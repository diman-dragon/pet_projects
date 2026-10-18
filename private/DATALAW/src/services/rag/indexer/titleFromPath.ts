export function titleFromPath(filePath: string): string {
  return (filePath.split(/[\\\/]/).pop() ?? filePath).replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
}
