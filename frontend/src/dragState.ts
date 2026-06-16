/** Fallback channel for in-app file drags.
 *
 * The file explorer also writes the dragged path into ``dataTransfer`` (see
 * ``DRAG_PATH_MIME``), but react-arborist's react-dnd backend can intercept
 * native drag data, so a drop target outside the tree may read nothing. This
 * module-level holder is set on drag start and read on drop as a guaranteed
 * path, independent of ``dataTransfer``.
 */

let draggedFilePath: string | null = null

export function setDraggedFilePath(path: string | null): void {
  draggedFilePath = path
}

export function getDraggedFilePath(): string | null {
  return draggedFilePath
}
