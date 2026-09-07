import { toneFor, humanize } from "../utils/format.js";

// A small status pill. `tone` can be passed explicitly; otherwise it is derived
// from the backend `value` string via toneFor().
export default function Badge({ value, tone, label, dot = true }) {
  const t = tone || toneFor(value);
  return (
    <span className={`badge tone-${t}`}>
      {dot && <span className="dot" />}
      {label ?? humanize(value)}
    </span>
  );
}
