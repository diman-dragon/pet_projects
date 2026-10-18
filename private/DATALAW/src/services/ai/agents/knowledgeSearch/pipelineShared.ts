export const MAX_RESULTS = 8;
export const STREAM_CHUNK_SIZE = 4;

export function streamText(
  text: string,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): void {
  for (let index = 0; index < text.length; index += STREAM_CHUNK_SIZE) {
    if (signal?.aborted) {
      break;
    }
    onToken(text.slice(index, index + STREAM_CHUNK_SIZE));
  }
}
