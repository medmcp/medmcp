# Stack jobs

A stack tool call runs in the core's backend pool, whose containers are removed
when they have been idle for two minutes, when another GPU stack needs the warm
slot, and when the core stops. Work that takes hours — an nnU-Net training — cannot
live there. A **job** is one approved tool call run to completion in a container of
its own, outside the pool.

## For someone running MedMCP

- A stack that supports jobs lists its job tools in `/api/stacks` (`jobs`). The
  approval dialog of such a call carries the label *Starts a job that keeps running
  after this chat*.
- A job keeps running when the chat is idle or closed, when other stacks (GPU ones
  included) are used, and when the core restarts or is updated. It ends when its
  work is done, when it is stopped, or when the machine or the Docker daemon goes
  down. Jobs are not restarted after a reboot; the stack's own resume tool
  (`resume_training`) continues the work.
- **One job at a time** on the machine by default. A second one is refused with a
  message, not queued: nothing starts later than the approval that asked for it.
  `MEDMCP_MAX_JOBS=2` allows more.
- **Jobs and other GPU tools share the GPU.** A running job does not occupy the
  pool's warm GPU slot and is never evicted to make room; GPU stack tools keep
  working next to it and compete with it (and with the local model) for memory.
  A job that runs out of GPU memory fails and can be resumed.
- Running jobs are shown in the Tool stacks window with a Stop button.
  `GET /api/jobs` lists running and finished jobs, `GET /api/jobs/{id}` adds the
  container's last output, `POST /api/jobs/{id}/stop` stops one (SIGTERM, 30 s to
  save its state, then SIGKILL), `DELETE /api/jobs/{id}` forgets a finished one.
  The newest 20 finished jobs are kept.
- A job's start is written into the provenance of the chat that approved it: stack,
  tool, arguments, the image reference, the image id that ran and its registry
  digest if it was pulled, the command line and the container name.
- `MEDMCP_STACK_JOBS=0` turns jobs off; stacks then run long work the way they did
  before (inside the pooled container, with its two-minute limit).

## For a stack author

Declare a job runner in the image's `org.medmcp.stack` label:

```json
{"name": "medmcp-nnunet", "gpu": true,
 "job": {"command": ["medmcp-nnunet-job", "call"],
         "tools": ["start_training", "resume_training"],
         "shm_size": "16g", "watch_sec": 15}}
```

- `command`: one to four plain words or paths, run through the image's own
  `ENTRYPOINT`. The core appends exactly two arguments: the tool's name and the
  approved call's arguments as one JSON object.
- `tools`: the only tools whose calls may become a job.
- `shm_size` (optional): `/dev/shm` for the job, if larger than the core's 8 GB
  floor (at most 64 GB). `watch_sec` (optional, 0–30, default 3): how long the core
  watches a new job; one that exits non-zero within that time fails the tool call,
  with its output.

The protocol:

1. When the core will act on a job request, the pooled stack container has
   `MEDMCP_JOBS=1` in its environment. Without it, do what you did before.
2. With it, a job tool validates its arguments, starts nothing, and returns a
   result whose top-level object has `"medmcp_job": {"name": "<short name>"}`.
   A failed call (`isError`) starts nothing.
3. The core starts `<command> <tool> '<arguments>'` in a detached container of the
   same image and replaces `medmcp_job` in the result with
   `{"started": true, "id": …, "status": …, "note": …}`. If it does not start the
   job (limit reached, image changed), the call fails with
   *MedMCP did not start the background job: …*. Tell the model, in the tool's
   render rules, to report a start only when `started` is true.
4. The job container gets `MEDMCP_JOB_ID`. It has the workspace at its own path
   and nothing else: no network (also when the stack was installed with egress),
   no stdin, and nobody reading its stdout except `docker logs`. Keep status and
   stop in files in the workspace, so the stack's ordinary tools can follow and
   stop the job from any pooled container. Handle SIGTERM: that is what Stop sends.

   It also gets `MEDMCP_JOB_PROGRESS`, a path in the workspace
   (`<workspace>/.medmcp/jobs/<id>.json`). A job that wants a progress line in the
   header writes `{"fraction": 0.41, "text": "epoch 41/100"}` there now and then
   (atomically: write a temporary file, then rename). Optional; the core keeps only
   the fraction, clamped to 0..1, and one line of at most 120 characters.
5. Validate the arguments again in the job: they are the approved ones, but the
   workspace may have changed since the tool looked.

## Why it is built this way

MedMCP assumes the model can be steered by what it reads, that the agent's shell
can reach the broker socket, and that nothing may run without the person's
approval. A job must not weaken any of that, and it adds one thing of its own: it
outlives the chat it was approved in.

- **Only an approved call can lead to a job.** There is no "start job" request on
  the broker socket. The broker spends the approval for `(stack, tool, arguments)`
  as for any call, runs the call in the pool, and only then hands the result to the
  job code. A connection that merely reaches the socket cannot get that far, and
  one approval yields at most one job.
- **The command line has no author but the image and the person.** Its fixed part
  comes from the image's label, read from the image when the job starts (not from
  the `stacks.d` manifest, which is a file) and run by image id, so what was read is
  what runs. Its variable part is the tool name and the arguments the ledger
  matched — what the dialog showed. The model contributes nothing beyond those
  arguments. The pooled stack's answer contributes only a display name: a stack
  container that was subverted while processing data (an nnU-Net checkpoint is a
  pickle) cannot turn someone's approved `predict` into a job, nor change what an
  approved `start_training` trains on.
- **The guard allows exactly this container.** The core starts containers through
  `medmcp-docker-guard`. A create that is named `medmcp-job-…` or carries an
  `org.medmcp.job` label must have no network, the workspace as its only mount, all
  capabilities dropped, `no-new-privileges`, a task limit, no restart policy, no
  entrypoint or user override, and a command that is the image's declared job
  command plus one declared tool and one argument. A bug in the core therefore
  cannot produce a long-lived container in any other shape under the job name.
  Images that themselves carry a job label are refused (containers inherit image
  labels), and a manifest may not name its container `medmcp-job-…`.
- **No state the agent can write.** A job is its container; the list is read from
  the daemon by label and name. That is also why jobs survive a core restart with
  no hand-over: the next core finds them.
- **No network, ever.** A stack that was installed with egress still gets
  `--network none` for its jobs. A container that holds the workspace for days is
  the worst place for a way out.
- **Visible and stoppable.** The dialog says the call starts a job; the job is
  listed with the chat that approved it, and anyone at the UI can stop it.

What a job does not change: the job container is as trusted as the stack image
itself, and has the workspace read-write, like every stack.
