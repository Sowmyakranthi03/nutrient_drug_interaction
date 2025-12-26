import React, { useState } from "react";
import { sendFeedback } from "../api.js";

function FoodRow({ item, drugs }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="row">
      <div className="rowLeft">
        <b>{item.food_name}</b>
        <div className="meta">
          <span>Score: {item.score.toFixed(2)}</span>
          <span>Key: {item.food_key}</span>
        </div>
        {open && (
          <div className="why">
            <div><b>Reasons:</b></div>
            {item.reasons?.length ? (
              <ul>
                {item.reasons.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            ) : (
              <div>No rule conflicts detected</div>
            )}
          </div>
        )}
      </div>

      <div className="rowRight">
        <button className="secondary" onClick={() => setOpen(!open)}>
          {open ? "Hide Why" : "Why?"}
        </button>
        <button onClick={() => sendFeedback(drugs, item.food_key, "safe")}>👍 Safe</button>
        <button onClick={() => sendFeedback(drugs, item.food_key, "unsafe")} className="danger">👎 Unsafe</button>
      </div>
    </div>
  );
}

export default function Results({ data }) {
  const [tab, setTab] = useState("safe");
  if (!data) return null;

  const list = tab === "safe" ? data.safe : data.unsafe;

  return (
    <div className="card">
      <h3>Results</h3>
      <div className="metaBlock">
        <div><b>Selected drugs:</b> {data.drugs.join(", ") || "-"}</div>
        <div><b>Avoid groups:</b> {data.avoid_groups.join(", ") || "-"}</div>
      </div>

      <div className="tabs">
        <button className={tab === "safe" ? "" : "secondary"} onClick={() => setTab("safe")}>
          ✅ Safe ({data.safe.length})
        </button>
        <button className={tab === "unsafe" ? "" : "secondary"} onClick={() => setTab("unsafe")}>
          ❌ Unsafe ({data.unsafe.length})
        </button>
      </div>

      <div>
        {list.map((item) => (
          <FoodRow key={item.food_key + tab} item={item} drugs={data.drugs} />
        ))}
      </div>
    </div>
  );
}
