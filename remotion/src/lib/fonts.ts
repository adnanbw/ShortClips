import { staticFile } from "remotion";

/**
 * CSS @font-face declaration for NotoSerif-Bold (bundled locally).
 * Use in components via: <style>{notoSerifFontFace}</style>
 */
export const NOTO_SERIF_FONT_FAMILY = "NotoSerif-Bold";

export const notoSerifFontFace = `
@font-face {
  font-family: '${NOTO_SERIF_FONT_FAMILY}';
  src: url('${staticFile("fonts/NotoSerif-Bold.ttf")}') format('truetype');
  font-weight: 700;
  font-style: normal;
}
`;

/**
 * Map of subtitle font families to their CSS-safe names.
 * These match the options available in SubtitleModal.jsx.
 */
export const SUBTITLE_FONTS: Record<string, string> = {
  Verdana: "Verdana, Geneva, sans-serif",
  Arial: "Arial, Helvetica, sans-serif",
  Impact: "Impact, Haettenschweiler, sans-serif",
  Helvetica: "Helvetica, Arial, sans-serif",
  Georgia: "Georgia, 'Times New Roman', serif",
  "Courier New": "'Courier New', Courier, monospace",
};

/**
 * Appended to every subtitle stack so a transcript in a non-Latin script does
 * not render as tofu boxes. Every family above is Latin-only, and a caption is
 * whatever language the speaker used. These come LAST, so Latin text keeps
 * rendering in exactly the font it always did — the browser only reaches them
 * for characters the chosen font has no glyph for.
 * The families must exist in the renderer image (render-service/Dockerfile
 * installs fonts-noto-core + fonts-noto-cjk).
 */
export const SCRIPT_FALLBACK_FONTS = [
  "'Noto Sans'",
  "'Noto Sans Devanagari'",
  "'Noto Naskh Arabic'",
  "'Noto Sans Hebrew'",
  "'Noto Sans Thai'",
  "'Noto Sans CJK SC'",
  "'Noto Sans CJK JP'",
  "'Noto Sans CJK KR'",
  "'Noto Color Emoji'",
].join(", ");

export function getFontStack(fontFamily: string): string {
  const base = SUBTITLE_FONTS[fontFamily] ?? fontFamily;
  return `${base}, ${SCRIPT_FALLBACK_FONTS}`;
}
