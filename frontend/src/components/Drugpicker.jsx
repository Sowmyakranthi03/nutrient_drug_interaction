import React, { useEffect, useState } from "react";
import { fetchDrugs } from "../api.js";

export default function DrugPicker({ selected, setSelected }) {
  const [q, setQ] = useState("");
  const [options, setOptions] = useState([]);

  useEffect(() => {
    let alive = true;
    fetchDrugs(q).then((d) => alive && setOptions(d));
    return () => (alive = false);
  }, [q]);

  function addDrug(name) {
    if (!name) return;
    const n = name.trim();
    if (!n) return;
    if (selected.includes(n)) return;
    setSelected([...selected, n]);
    setQ("");
  }

  return (
    <div className="card">
      <h3>Choose Drug(s)</h3>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Type drug name…"
        list="drug-list"
      />
      <datalist id="drug-list">
        {options.map((d) => (
          <option key={d} value={d} />
        ))}
      </datalist>

      <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
        <button onClick={() => addDrug(q)}>Add</button>
        <button onClick={() => setSelected([])} className="secondary">
          Clear
        </button>
      </div>

      <div className="chips">
        {selected.map((d) => (
          <span key={d} className="chip" onClick={() => setSelected(selected.filter(x => x !== d))}>
            {d} ✕
          </span>
        ))}
      </div>
      <small>Tip: click a chip to remove</small>
    </div>
  );
}
