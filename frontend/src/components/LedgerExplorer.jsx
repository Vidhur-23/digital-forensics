import { Fragment, useEffect, useState, useCallback } from "react";
import Panel from "./Panel.jsx";
import Badge from "./Badge.jsx";
import {
  getChainStats,
  verifyChain,
  listBlocks,
  tamperChain,
  restoreChain,
} from "../api/client.js";
import { humanize } from "../utils/format.js";

// Block explorer for the tamper-evident chain. Read-only: it renders exactly
// what the backend /api/chain endpoints return. Blocks carry only fingerprints
// (SHA-256 hashes) — never PII — so this whole view is safe to show.

// Short 0x-style hash for compact display; full value stays in the title.
function short(hash, head = 10, tail = 6) {
  if (!hash) return "—";
  if (hash.length <= head + tail) return hash;
  return `${hash.slice(0, head)}…${hash.slice(-tail)}`;
}

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts.endsWith?.("Z") ? ts : `${ts}Z`);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleString();
}

function TransactionRow({ tx }) {
  const p = tx.payload || {};
  return (
    <div className="tx-row">
      <div className="tx-head">
        <span className="pill">{humanize(tx.action)}</span>
        {p.risk_level && <Badge value={p.risk_level} label={`${p.risk_level} risk`} />}
      </div>
      <dl className="kv">
        <dt>subject</dt>
        <dd>
          {tx.subject_type}
          {tx.subject_id ? ` · ${tx.subject_id}` : ""}
        </dd>
        {p.result_sha256 && (
          <>
            <dt>result hash</dt>
            <dd title={p.result_sha256}>{short(p.result_sha256)}</dd>
          </>
        )}
        {Array.isArray(p.evidence) &&
          p.evidence.map((e, i) => (
            <Fragment key={`${e.kind}-${i}`}>
              <dt>{e.kind} hash</dt>
              <dd title={e.sha256}>{short(e.sha256)}</dd>
            </Fragment>
          ))}
        {p.recommendation && (
          <>
            <dt>recommendation</dt>
            <dd>{humanize(p.recommendation)}</dd>
          </>
        )}
        <dt>tx hash</dt>
        <dd title={tx.tx_hash}>{short(tx.tx_hash)}</dd>
      </dl>
    </div>
  );
}

function BlockCard({ block, broken }) {
  const isGenesis = block.index === 0;
  return (
    <div className={`block-card ${broken ? "block-broken" : ""}`}>
      <div className="block-head">
        <div className="block-id">
          <span className="block-hash-badge">⛓</span>
          <span className="block-height">Block #{block.index}</span>
          {isGenesis && <span className="pill">genesis</span>}
        </div>
        {broken ? (
          <span className="pill pill-bad" title="verification failed here">
            ✗ tampered
          </span>
        ) : (
          <span className="pill" title="proof-of-work difficulty">
            difficulty {block.difficulty}
          </span>
        )}
      </div>

      <dl className="kv block-kv">
        <dt>hash</dt>
        <dd className="hash-strong" title={block.hash}>
          {short(block.hash, 14, 8)}
        </dd>
        <dt>prev</dt>
        <dd title={block.prev_hash}>{short(block.prev_hash, 14, 8)}</dd>
        <dt>merkle root</dt>
        <dd title={block.merkle_root}>{short(block.merkle_root, 14, 8)}</dd>
        <dt>nonce</dt>
        <dd>{block.nonce}</dd>
        <dt>mined</dt>
        <dd>{fmtTime(block.timestamp)}</dd>
        {block.miner && (
          <>
            <dt>miner</dt>
            <dd>{block.miner}</dd>
          </>
        )}
      </dl>

      {block.transactions?.length > 0 && (
        <div className="tx-list">
          <div className="section-note">
            {block.transactions.length} transaction
            {block.transactions.length === 1 ? "" : "s"}
          </div>
          {block.transactions.map((tx) => (
            <TransactionRow key={tx.tx_hash} tx={tx} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function LedgerExplorer() {
  const [stats, setStats] = useState(null);
  const [blocks, setBlocks] = useState([]);
  const [verify, setVerify] = useState(null);
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [demoBusy, setDemoBusy] = useState(false);
  const [demoMsg, setDemoMsg] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, b] = await Promise.all([getChainStats(), listBlocks(100)]);
      setStats(s);
      setBlocks(b);
    } catch (e) {
      setError(e.message || "Could not load the chain.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleVerify() {
    setVerifying(true);
    setError(null);
    try {
      setVerify(await verifyChain());
    } catch (e) {
      setError(e.message || "Verification failed.");
    } finally {
      setVerifying(false);
    }
  }

  async function handleTamper() {
    setDemoBusy(true);
    setError(null);
    try {
      const res = await tamperChain();
      if (!res.tampered) {
        setDemoMsg({ tone: "warn", text: res.reason || "Nothing to tamper." });
      } else {
        setDemoMsg({
          tone: "warn",
          text: `Simulated attack: block #${res.block_index} "${res.field}" changed ${res.before} → ${res.after}. Now verify the chain.`,
        });
        await load();
        setVerify(await verifyChain()); // show it caught the tamper
      }
    } catch (e) {
      setError(e.message || "Tamper demo failed.");
    } finally {
      setDemoBusy(false);
    }
  }

  async function handleRestore() {
    setDemoBusy(true);
    setError(null);
    try {
      const res = await restoreChain();
      setDemoMsg({
        tone: "good",
        text: `Restored ${res.restored} transaction(s) to their sealed values.`,
      });
      await load();
      setVerify(await verifyChain());
    } catch (e) {
      setError(e.message || "Restore failed.");
    } finally {
      setDemoBusy(false);
    }
  }

  const brokenAt = verify && !verify.ok ? verify.broken_at : null;

  return (
    <>
      <Panel title="How the chain works" note="A permissioned, tamper-evident blockchain — running inside this app, no external network.">
        <ol className="how-list">
          <li>
            Every screening <strong>mines a new block</strong> that commits only
            the record's <strong>SHA-256 fingerprint</strong> and risk metadata —
            never the document, PII or biometrics.
          </li>
          <li>
            Each block links to the previous block's hash and is sealed with a
            <strong> proof-of-work nonce</strong>, forming an unbroken chain.
          </li>
          <li>
            Change, delete or reorder <em>anything</em> and every later hash stops
            matching — <strong>verification fails and pinpoints the block</strong>.
          </li>
          <li>
            The confidential evidence stays off-chain, encrypted and{" "}
            <strong>deletable</strong> (DPDP / GDPR) — erasing it just orphans a
            harmless hash.
          </li>
        </ol>
      </Panel>

      <Panel
        title="Chain of custody"
        right={
          <span className="pill">
            {stats ? `height ${stats.height}` : "…"}
          </span>
        }
        note="Every screening mines a block committing the record's fingerprint (SHA-256) — never the document, PII or biometrics. Any edit, deletion or reorder breaks the chain and fails verification."
      >
        <div className="chain-stats">
          <div className="chain-stat">
            <span className="chain-stat-num">{stats?.blocks ?? "—"}</span>
            <span className="chain-stat-label">blocks</span>
          </div>
          <div className="chain-stat">
            <span className="chain-stat-num">{stats?.transactions ?? "—"}</span>
            <span className="chain-stat-label">transactions</span>
          </div>
          <div className="chain-stat">
            <span className="chain-stat-num">{stats?.difficulty ?? "—"}</span>
            <span className="chain-stat-label">difficulty</span>
          </div>
          <div className="chain-stat chain-stat-wide">
            <span className="chain-stat-num mono" title={stats?.head || ""}>
              {short(stats?.head, 12, 8)}
            </span>
            <span className="chain-stat-label">head hash</span>
          </div>
        </div>

        <div className="chain-actions">
          <button className="btn" onClick={handleVerify} disabled={verifying}>
            {verifying ? "Verifying…" : "Verify chain integrity"}
          </button>
          <button className="btn btn-ghost" onClick={load} disabled={loading}>
            Refresh
          </button>
        </div>

        {verify && (
          <div className={`banner ${verify.ok ? "tone-good" : "tone-bad"}`}>
            {verify.ok
              ? `✓ Chain intact — ${verify.blocks} blocks, ${verify.transactions} transactions verified. Head ${short(
                  verify.head,
                  12,
                  8
                )}.`
              : `✗ Chain BROKEN at block #${verify.broken_at}: ${verify.reason}`}
          </div>
        )}

        {error && <div className="banner tone-bad">{error}</div>}
      </Panel>

      <Panel
        title="Live demonstration"
        note="Prove it works: simulate an attacker editing a sealed block, watch verification catch it, then restore. (Demo-only endpoints.)"
      >
        <div className="chain-actions">
          <button
            className="btn btn-danger"
            onClick={handleTamper}
            disabled={demoBusy}
          >
            {demoBusy ? "Working…" : "① Simulate tamper"}
          </button>
          <button className="btn" onClick={handleVerify} disabled={verifying}>
            ② Verify
          </button>
          <button
            className="btn btn-ghost"
            onClick={handleRestore}
            disabled={demoBusy}
          >
            ③ Restore
          </button>
        </div>
        {demoMsg && (
          <div className={`banner tone-${demoMsg.tone}`}>{demoMsg.text}</div>
        )}
      </Panel>

      {loading ? (
        <Panel title="Blocks">
          <div className="empty">Loading the chain…</div>
        </Panel>
      ) : blocks.length === 0 ? (
        <Panel title="Blocks">
          <div className="empty">
            No blocks yet. Run a screening to mine the first block.
          </div>
        </Panel>
      ) : (
        <div className="stack chain-blocks">
          {blocks.map((b) => (
            <BlockCard key={b.hash} block={b} broken={brokenAt === b.index} />
          ))}
        </div>
      )}
    </>
  );
}
