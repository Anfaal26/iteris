/**
 * Dataset / modality auto-detection for the workspace upload flow.
 *
 * The redesign removes the manual CAMUS/BRISC toggle: the dataset is inferred
 * from the uploaded file and surfaced as a read-only chip, never as a control
 * the user sets. Detection order (most reliable first):
 *   1. DICOM  — "DICM" magic at byte 128, then the Modality tag (US → CAMUS, MR → BRISC).
 *   2. NIfTI  — "n+1"/"ni1" magic at byte 344 (volumetric → MRI → BRISC).
 *   3. Filename hints — camus/echo/heart vs brisc/brain/tumour/mri.
 *   4. Pixel dimensions — sector-shaped echo tends portrait; MRI slices near-square.
 *
 * Everything runs in the browser off the raw File; nothing is uploaded to detect.
 */
import type { DatasetId, Modality } from '@/api/contract';

/** Outcome of a detection pass. */
export interface DetectionResult {
  dataset: DatasetId;
  modality: Modality;
  /** Short human label for the read-only chip, e.g. "cardiac echo (CAMUS)". */
  label: string;
  /** How the decision was reached — powers a subtle provenance hint. */
  source: 'dicom' | 'nifti' | 'filename' | 'dimensions';
  /** Low means the dimension fallback guessed; high means metadata/filename was explicit. */
  confidence: 'high' | 'low';
}

const CAMUS_RESULT = (
  source: DetectionResult['source'],
  confidence: DetectionResult['confidence'],
): DetectionResult => ({
  dataset: 'camus',
  modality: 'ultrasound',
  label: 'cardiac echo (CAMUS)',
  source,
  confidence,
});

const BRISC_RESULT = (
  source: DetectionResult['source'],
  confidence: DetectionResult['confidence'],
): DetectionResult => ({
  dataset: 'brisc',
  modality: 'mri',
  label: 'brain MRI (BRISC)',
  source,
  confidence,
});

/** Read the leading bytes of a file without loading the whole thing. */
async function headBytes(file: File, count: number): Promise<Uint8Array> {
  const slice = file.slice(0, count);
  return new Uint8Array(await slice.arrayBuffer());
}

/** ASCII decode of a byte range, for magic-number checks. */
function ascii(bytes: Uint8Array, start: number, end: number): string {
  let out = '';
  for (let i = start; i < end && i < bytes.length; i += 1) {
    out += String.fromCharCode(bytes[i]);
  }
  return out;
}

/** DICOM files carry the literal "DICM" at byte offset 128. */
function isDicom(bytes: Uint8Array): boolean {
  return bytes.length >= 132 && ascii(bytes, 128, 132) === 'DICM';
}

/**
 * NIfTI-1 stores the magic "n+1\0" (single-file) or "ni1\0" (header/image pair)
 * at byte offset 344. Gzipped `.nii.gz` won't match here — filename covers that.
 */
function isNifti(bytes: Uint8Array): boolean {
  const magic = ascii(bytes, 344, 347);
  return magic === 'n+1' || magic === 'ni1';
}

/**
 * Best-effort DICOM Modality (0008,0060) sniff. We don't parse the full data
 * set — we scan the first chunk for the "US"/"MR" value that follows the tag
 * bytes. Returns null when it can't tell, so callers fall back to dimensions.
 */
function dicomModality(bytes: Uint8Array): DatasetId | null {
  // Tag (0008,0060) little-endian on the wire: 08 00 60 00.
  for (let i = 0; i + 12 < bytes.length; i += 1) {
    if (
      bytes[i] === 0x08 &&
      bytes[i + 1] === 0x00 &&
      bytes[i + 2] === 0x60 &&
      bytes[i + 3] === 0x00
    ) {
      // Value sits a few bytes on (VR + length vary by transfer syntax); scan a
      // small window for the two-letter modality code.
      const window = ascii(bytes, i + 4, i + 20);
      if (window.includes('US')) return 'camus';
      if (window.includes('MR')) return 'brisc';
      return null;
    }
  }
  return null;
}

/** Filename keyword hints — explicit enough to trust over the dimension guess. */
function detectFromFilename(name: string): DetectionResult | null {
  const n = name.toLowerCase();
  if (/(camus|echo|cardiac|heart|us_)/.test(n)) return CAMUS_RESULT('filename', 'high');
  if (/(brisc|brain|tumou?r|glioma|meningioma|pituitary|mri|t1|t2|flair)/.test(n)) {
    return BRISC_RESULT('filename', 'high');
  }
  return null;
}

/** Decode intrinsic pixel dimensions from a raster file (PNG/JPG). */
async function imageDimensions(
  file: File,
): Promise<{ width: number; height: number } | null> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    const loaded = new Promise<boolean>((resolve) => {
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
    });
    img.src = url;
    const ok = await loaded;
    if (!ok || !img.naturalWidth) return null;
    return { width: img.naturalWidth, height: img.naturalHeight };
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Detect the dataset + modality for an uploaded scan. Always resolves — the
 * dimension fallback picks a best guess (flagged `confidence: 'low'`) so the UI
 * can show the chip with a softer, "looks like…" affordance when unsure.
 */
export async function detectDataset(file: File): Promise<DetectionResult> {
  // 1 + 2 — metadata magic numbers (need only the header bytes).
  try {
    const head = await headBytes(file, 512);
    if (isDicom(head)) {
      const mod = dicomModality(head);
      if (mod === 'camus') return CAMUS_RESULT('dicom', 'high');
      if (mod === 'brisc') return BRISC_RESULT('dicom', 'high');
      // DICOM but modality unreadable — keep going to filename/dimensions.
    }
    if (isNifti(head)) return BRISC_RESULT('nifti', 'high');
  } catch {
    // Unreadable header — fall through to filename/dimensions.
  }

  // `.nii.gz` (gzip magic 1f 8b) is always the volumetric MRI path.
  if (/\.nii\.gz$/i.test(file.name)) return BRISC_RESULT('nifti', 'high');

  // 3 — filename keywords.
  const byName = detectFromFilename(file.name);
  if (byName) return byName;

  // 4 — pixel-dimension heuristic. Sector-shaped echo frames skew portrait;
  // axial MRI slices are typically square. Weak signal → low confidence.
  const dims = await imageDimensions(file);
  if (dims && dims.height > 0) {
    const aspect = dims.width / dims.height;
    return aspect < 0.9
      ? CAMUS_RESULT('dimensions', 'low')
      : BRISC_RESULT('dimensions', 'low');
  }

  // Nothing decisive — default to CAMUS (the primary DRL target) at low confidence.
  return CAMUS_RESULT('dimensions', 'low');
}
