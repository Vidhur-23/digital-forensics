// App chrome: header (title row + live pipeline nav) + footer.
export default function AppShell({ nav, children }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header-top">
          <div className="brand">
            <div className="logo">🛂</div>
            <div>
              <h1 style={{ fontSize: 17 }}>Document Screening</h1>
              <div className="subtitle">
                Intelligence Layer · rules · forensics · biometrics · risk
              </div>
            </div>
          </div>
          <div className="subtitle">SIH26188</div>
        </div>
        {nav}
      </header>

      <main className="app-main">{children}</main>

      <footer className="app-footer">
        Evidence-based decision support. Findings are evidence, not verdicts —
        the human officer makes the final decision.
      </footer>
    </div>
  );
}
