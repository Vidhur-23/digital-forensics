// Generic titled panel wrapper.
export default function Panel({ title, right, children, note }) {
  return (
    <section className="panel">
      {(title || right) && (
        <div className="panel-head">
          {title ? <h3>{title}</h3> : <span />}
          {right}
        </div>
      )}
      <div className="panel-body">
        {note && <div className="section-note">{note}</div>}
        {children}
      </div>
    </section>
  );
}
