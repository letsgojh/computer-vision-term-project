// 촬영 가이드 도식용 SVG 조각들. 각 함수는 SVG 문자열을 돌려준다.
const S = {};

S.svg = (w, h, body, cls = "") =>
  `<svg class="frame ${cls}" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`;

// ---- 배경 ----
S.desk = (w, h) => {
  let s = `<rect width="${w}" height="${h}" fill="#e8d8bd"/>`;
  for (let y = 18; y < h; y += 34) {
    s += `<path d="M0 ${y} C ${w * 0.3} ${y - 6}, ${w * 0.6} ${y + 7}, ${w} ${y - 2}" stroke="#dbc49f" stroke-width="2" fill="none"/>`;
  }
  return s;
};
S.floor = (w, h) => {
  let s = `<rect width="${w}" height="${h}" fill="#cfd3d6"/>`;
  for (let x = 0; x < w; x += 110) s += `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="#bcc1c5" stroke-width="3"/>`;
  for (let y = 0; y < h; y += 110) s += `<line x1="0" y1="${y}" x2="${w}" y2="${y}" stroke="#bcc1c5" stroke-width="3"/>`;
  return s;
};
S.bed = (w, h) => {
  let s = `<rect width="${w}" height="${h}" fill="#dfe8f3"/>`;
  for (let i = 0; i < 5; i++) {
    const y = 40 + i * (h / 5);
    s += `<path d="M-20 ${y} Q ${w * 0.35} ${y + 30} ${w * 0.6} ${y + 5} T ${w + 20} ${y + 15}" stroke="#c9d6e6" stroke-width="6" fill="none"/>`;
  }
  return s;
};
S.wall = (w, h) =>
  `<rect width="${w}" height="${h}" fill="#eceae4"/><rect y="${h * 0.86}" width="${w}" height="${h * 0.14}" fill="#c9b08a"/>`;

// 조명: natural(그대로) / fluorescent(푸른 기) / dark(어둡게)
S.light = (w, h, type) =>
  type === "fluorescent" ? `<rect width="${w}" height="${h}" fill="#bcd7ff" opacity="0.28"/>`
  : type === "dark" ? `<rect width="${w}" height="${h}" fill="#141821" opacity="0.5"/>`
  : "";

// ---- 물건 ----
// 위에서 본 마우스. (x, y)는 중심, k는 배율 (k=1이면 약 200x300)
S.mouse = (x, y, k = 1, rot = 0) => `
<g transform="translate(${x} ${y}) rotate(${rot}) scale(${k})">
  <ellipse cx="10" cy="14" rx="104" ry="152" fill="#000" opacity="0.12"/>
  <path d="M0,-150 C80,-150 100,-90 100,0 C100,100 70,150 0,150 C-70,150 -100,100 -100,0 C-100,-90 -80,-150 0,-150 Z" fill="#3b4148"/>
  <path d="M0,-150 L0,-42" stroke="#23272c" stroke-width="4"/>
  <path d="M-97,-44 Q0,-30 97,-44" stroke="#23272c" stroke-width="4" fill="none"/>
  <rect x="-9" y="-122" width="18" height="48" rx="9" fill="#8b929b"/>
  <path d="M-60,-120 C-40,-140 -20,-144 -6,-145" stroke="#5c646d" stroke-width="5" fill="none" stroke-linecap="round"/>
</g>`;

// 정면에서 본 냉장고. (x, y)는 왼쪽 위
S.fridge = (x, y, w = 230, h = 380) => `
<g transform="translate(${x} ${y})">
  <rect x="6" y="8" width="${w}" height="${h}" rx="14" fill="#000" opacity="0.1"/>
  <rect width="${w}" height="${h}" rx="14" fill="#f4f6f8" stroke="#aab2bb" stroke-width="3"/>
  <line x1="0" y1="${h * 0.38}" x2="${w}" y2="${h * 0.38}" stroke="#aab2bb" stroke-width="3"/>
  <rect x="${w - 30}" y="${h * 0.12}" width="10" height="${h * 0.18}" rx="5" fill="#8b929b"/>
  <rect x="${w - 30}" y="${h * 0.46}" width="10" height="${h * 0.26}" rx="5" fill="#8b929b"/>
</g>`;

// ---- 메모 ----
// paper: postit / a4 / receipt / notebook. (x, y)는 중심, s는 한 변 길이
S.note = (x, y, s = 130, rot = -4, lines = ["2026.11.17"], opt = {}) => {
  const paper = opt.paper || "postit";
  const fs = opt.fontSize || s * 0.17;
  const font = opt.font || "Noteworthy, 'Bradley Hand', cursive";
  const ink = opt.ink || "#1d3a8a";
  let w = s, h = s, fill = opt.color || "#fbe36a", deco = "";
  if (paper === "a4") { w = s * 1.25; h = s * 0.9; fill = "#ffffff"; }
  if (paper === "receipt") { w = s * 0.75; h = s * 1.2; fill = "#fbfbf8";
    deco = Array.from({ length: 5 }, (_, i) => `<line x1="${-w / 2 + 10}" y1="${-h / 2 + 14 + i * 9}" x2="${w / 2 - 10 - (i % 2) * 18}" y2="${-h / 2 + 14 + i * 9}" stroke="#c4c4c4" stroke-width="2"/>`).join(""); }
  if (paper === "notebook") { w = s * 1.1; h = s; fill = "#ffffff";
    deco = Array.from({ length: 6 }, (_, i) => `<line x1="${-w / 2}" y1="${-h / 2 + 18 + i * (h / 6.5)}" x2="${w / 2}" y2="${-h / 2 + 18 + i * (h / 6.5)}" stroke="#bcd3ee" stroke-width="1.5"/>`).join("")
      + `<line x1="${-w / 2 + 16}" y1="${-h / 2}" x2="${-w / 2 + 16}" y2="${h / 2}" stroke="#f0a3a3" stroke-width="1.5"/>`; }
  const ty0 = paper === "receipt" ? h * 0.2 : -((lines.length - 1) * fs * 1.25) / 2 + fs * 0.35;
  const text = lines.map((t, i) =>
    `<text x="0" y="${ty0 + i * fs * 1.25}" text-anchor="middle" font-family="${font}" font-size="${fs}" fill="${ink}" ${opt.textStyle || ""}>${t}</text>`).join("");
  return `
<g transform="translate(${x} ${y}) rotate(${rot})">
  <rect x="${-w / 2 + 4}" y="${-h / 2 + 6}" width="${w}" height="${h}" fill="#000" opacity="0.13"/>
  <rect x="${-w / 2}" y="${-h / 2}" width="${w}" height="${h}" fill="${fill}" stroke="${paper === "postit" ? "none" : "#d6d6d6"}"/>
  ${deco}${text}
</g>`;
};

// ---- 표시 ----
S.num = (x, y, n) =>
  `<circle cx="${x}" cy="${y}" r="17" fill="#1f2328"/><text x="${x}" y="${y + 6}" text-anchor="middle" font-family="Pretendard Variable, sans-serif" font-size="18" font-weight="700" fill="#fff">${n}</text>`;
S.leader = (x1, y1, x2, y2) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#1f2328" stroke-width="2.5"/><circle cx="${x2}" cy="${y2}" r="4" fill="#1f2328"/>`;
S.dashbox = (x, y, w, h, color = "#1f2328") =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="2.5" stroke-dasharray="9 7"/>`;
S.cross = (x, y, r = 26, color = "#cf222e") =>
  `<path d="M${x - r} ${y - r} L${x + r} ${y + r} M${x + r} ${y - r} L${x - r} ${y + r}" stroke="${color}" stroke-width="7" stroke-linecap="round"/>`;

// 사진 한 칸 + 아래 표시 (kind: good / bad)
S.panel = (svg, kind, cap) =>
  `<div class="panel">${svg}<div class="label ${kind}">${kind === "good" ? "좋은 예" : "나쁜 예"}</div><div class="cap">${cap}</div></div>`;

S.fridgeAngled = (x, y, w = 80, h = 158, side = 34) => `
<g transform="translate(${x} ${y})">
  <polygon points="${w},0 ${w + side},12 ${w + side},${h - 12} ${w},${h}" fill="#d9dee3" stroke="#aab2bb" stroke-width="3"/>
  <rect width="${w}" height="${h}" rx="10" fill="#f4f6f8" stroke="#aab2bb" stroke-width="3"/>
  <line x1="0" y1="${h * 0.38}" x2="${w}" y2="${h * 0.38}" stroke="#aab2bb" stroke-width="3"/>
  <rect x="${w - 22}" y="${h * 0.12}" width="8" height="${h * 0.18}" rx="4" fill="#8b929b"/>
  <rect x="${w - 22}" y="${h * 0.46}" width="8" height="${h * 0.26}" rx="4" fill="#8b929b"/>
</g>`;
