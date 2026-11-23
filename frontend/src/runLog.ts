/** A replay log's steps by batch item, in log order. A step without an item
 *  index belongs to item 0 (a single run). */
export function groupByItem<T extends { item?: number | null }>(log: readonly T[]): Map<number, T[]> {
  const byItem = new Map<number, T[]>()
  for (const s of log) {
    const key = s.item ?? 0
    const bucket = byItem.get(key)
    if (bucket) bucket.push(s)
    else byItem.set(key, [s])
  }
  return byItem
}
