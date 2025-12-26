const BASE = "http://127.0.0.1:8000";

export async function fetchDrugs(q) {
  const url = q ? `${BASE}/drugs?q=${encodeURIComponent(q)}` : `${BASE}/drugs`;
  const r = await fetch(url);
  return (await r.json()).drugs || [];
}

export async function recommend(drugs, top_k = 50) {
  const r = await fetch(`${BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drugs, top_k }),
  });
  return await r.json();
}

export async function sendFeedback(drugs, food_key, user_label) {
  await fetch(`${BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drugs, food_key, user_label }),
  });
}
