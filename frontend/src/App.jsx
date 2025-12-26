import React, { useEffect, useMemo, useState } from "react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";


function Badge({ children }) {
  return <span className="badge">{children}</span>;
}

function Card({ children }) {
  return <div className="card">{children}</div>;
}

export default function App() {
  const [drugOptions, setDrugOptions] = useState([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState([]);
  const [topK, setTopK] = useState(30);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  // theme
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "light");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    fetch(`${API}/drugs`)
      .then((r) => r.json())
      .then((d) => setDrugOptions(d.drugs || []))
      .catch(() => setDrugOptions([]));
  }, []);

  const filteredOptions = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return drugOptions.slice(0, 30);
    return drugOptions.filter((x) => x.toLowerCase().includes(q)).slice(0, 50);
  }, [drugOptions, query]);

  const addDrug = (d) => {
    const val = (d || "").trim();
    if (!val) return;
    if (selected.includes(val)) return;
    setSelected((prev) => [...prev, val]);
    setQuery("");
  };

  const removeDrug = (d) => setSelected((prev) => prev.filter((x) => x !== d));

  const runRecommend = async () => {
    setErr("");
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drugs: selected, top_k: Number(topK) || 30 }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setErr(e.message || "Failed");
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (food_key, vote) => {
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drugs: selected, food_key, vote }),
      });
      runRecommend();
    } catch {}
  };

  return (
    <div className="page">
      <div className="container">
        <div className="header">
          <div>
            <h1 className="title">Nutrient–Drug Interaction DSS</h1>
            <p className="subtitle">
              Select drugs → get Safe & Unsafe foods (rule + feedback ranking).
            </p>
          </div>

          <button
            className="btn btn-ghost"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title="Toggle dark mode"
          >
            {theme === "dark" ? "🌙 Dark" : "☀️ Light"}
          </button>
        </div>

        <div className="grid-2">
          <Card>
            <h3 className="h3">1) Choose drugs</h3>

            <div className="row">
              <input
                className="input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Type drug name (fuzzy match supported)…"
              />
              <button className="btn btn-primary" onClick={() => addDrug(query)}>
                Add
              </button>
            </div>

            <div className="chips">
              {filteredOptions.map((d) => (
                <button key={d} className="chip" onClick={() => addDrug(d)} title="Click to add">
                  {d}
                </button>
              ))}
            </div>

            <div className="selected">
              <h4 className="h4">Selected</h4>

              {selected.length === 0 ? (
                <div className="muted">No drugs selected yet.</div>
              ) : (
                <div className="selectedWrap">
                  {selected.map((d) => (
                    <span key={d} className="selectedItem">
                      <Badge>{d}</Badge>
                      <button className="remove" onClick={() => removeDrug(d)} title="Remove">
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </Card>

          <Card>
            <h3 className="h3">2) Run</h3>

            <label className="label">Top K foods</label>
            <input
              className="input"
              type="number"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              min={1}
              max={200}
            />

            <button
              className="btn btn-primary btn-block"
              onClick={runRecommend}
              disabled={loading || selected.length === 0}
            >
              {loading ? "Running…" : "Recommend Foods"}
            </button>

            {err && <div className="error">{err}</div>}

            {result && (
              <div className="meta">
                <div>
                  Avoid groups:{" "}
                  <b>{(result.avoid_groups || []).join(", ") || "None"}</b>
                </div>
                <div>
                  Returned: <b>{result.counts?.safe}</b> safe /{" "}
                  <b>{result.counts?.unsafe}</b> unsafe
                </div>
              </div>
            )}
          </Card>
        </div>

        {result && (
          <div className="grid-2 mt">
            <Card>
              <h3 className="h3">✅ Safe foods</h3>

              {(result.safe || []).map((f) => (
                <div key={f.food_key} className="foodRow">
                  <div className="foodTop">
                    <div>
                      <div className="foodName">{f.food_name}</div>
                      <div className="foodMeta">
                        Key: {f.food_key} • Class: {f.classification} • Score:{" "}
                        {Number(f.score).toFixed(2)}
                      </div>
                    </div>

                    <div className="votes">
                      <button className="miniBtn" onClick={() => sendFeedback(f.food_key, "up")}>
                        👍
                      </button>
                      <button className="miniBtn" onClick={() => sendFeedback(f.food_key, "down")}>
                        👎
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </Card>

            <Card>
              <h3 className="h3">⛔ Unsafe foods</h3>

              {(result.unsafe || []).map((f) => (
                <div key={f.food_key} className="foodRow">
                  <div className="foodName">{f.food_name}</div>
                  <div className="foodMeta">
                    Key: {f.food_key} • Class: {f.classification} • Score:{" "}
                    {Number(f.score).toFixed(2)}
                  </div>
                  <div className="reasons">
                    Reasons: {(f.reasons || []).join(", ")}
                  </div>
                </div>
              ))}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
