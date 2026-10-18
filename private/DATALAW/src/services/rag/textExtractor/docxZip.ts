import { inflateRawBytes } from './inflate';

const EOCD_SIGNATURE = 0x06054b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;

function readUint16(view: DataView, offset: number): number {
  return view.getUint16(offset, true);
}

function readUint32(view: DataView, offset: number): number {
  return view.getUint32(offset, true);
}

function findEndOfCentralDirectory(bytes: Uint8Array, view: DataView): number {
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65558); i -= 1) {
    if (readUint32(view, i) === EOCD_SIGNATURE) return i;
  }

  throw new Error('EOCD не найден: файл не похож на ZIP/DOCX.');
}

export async function readDocxDocumentXml(bytes: Uint8Array): Promise<string> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocdOffset = findEndOfCentralDirectory(bytes, view);
  const centralDirectoryOffset = readUint32(view, eocdOffset + 16);
  const centralDirectorySize = readUint32(view, eocdOffset + 12);
  const decoder = new TextDecoder('utf-8');

  let position = centralDirectoryOffset;
  while (
    position + 46 <= centralDirectoryOffset + centralDirectorySize &&
    position + 46 <= bytes.length
  ) {
    if (readUint32(view, position) !== CENTRAL_DIRECTORY_SIGNATURE) break;

    const compressionMethod = readUint16(view, position + 10);
    const compressedSize = readUint32(view, position + 20);
    const fileNameLength = readUint16(view, position + 28);
    const extraLength = readUint16(view, position + 30);
    const commentLength = readUint16(view, position + 32);
    const localHeaderOffset = readUint32(view, position + 42);
    const fileName = decoder.decode(bytes.slice(position + 46, position + 46 + fileNameLength));

    if (fileName === 'word/document.xml') {
      const localFileNameLength = readUint16(view, localHeaderOffset + 26);
      const localExtraLength = readUint16(view, localHeaderOffset + 28);
      const dataStart = localHeaderOffset + 30 + localFileNameLength + localExtraLength;
      const compressedData = bytes.slice(dataStart, dataStart + compressedSize);

      if (compressionMethod === 0) return decoder.decode(compressedData);
      if (compressionMethod === 8) return decoder.decode(await inflateRawBytes(compressedData));

      throw new Error(`Неподдерживаемый ZIP-метод сжатия: ${compressionMethod}`);
    }

    position += 46 + fileNameLength + extraLength + commentLength;
  }

  throw new Error('word/document.xml не найден в DOCX.');
}
