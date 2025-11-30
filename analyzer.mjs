import dotenv from "dotenv";
dotenv.config();

// =============== CONFIG ===================
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HF_URL = "https://router.huggingface.co";   // NEW router URL
// ==========================================

if (!HF_API_KEY) {
  console.error("❌ ERROR: Missing Hugging Face API key in .env file.");
  process.exit(1);
}

const SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest";
const INTENT_MODEL = "facebook/bart-large-mnli";

const INTENT_LABELS = [
  "greeting",
  "product_query",
  "order_status",
  "refund",
  "cancel_order",
  "password_reset",
  "account_login",
  "bug_report",
  "feature_request",
  "complaint",
  "praise",
  "question"
];

// ==========================================================
// GENERIC MODEL CALL  (Updated for new URL)
// ==========================================================
async function callModel(model, payload) {
  const url = `${HF_URL}/hf-inference/models/${model}`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} → ${err}`);
  }

  return res.json();
}

// ==========================================================
// SENTIMENT ANALYSIS
// ==========================================================
async function getSentiment(text) {
  const data = await callModel(SENTIMENT_MODEL, { inputs: text });

  const arr = Array.isArray(data) && Array.isArray(data[0]) ? data[0] : data;
  const top = arr.reduce((best, cur) => (cur.score > best.score ? cur : best), arr[0]);

  return { label: top.label, score: top.score };
}

// ==========================================================
// INTENT DETECTION (Zero-Shot)
// ==========================================================
async function getIntent(text) {
  const data = await callModel(INTENT_MODEL, {
    inputs: text,
    parameters: {
      candidate_labels: INTENT_LABELS,
      multi_label: true
    }
  });

  const labels = data.labels || [];
  const scores = data.scores || [];

  const ranked = labels
    .map((l, i) => ({ label: l, score: scores[i] }))
    .sort((a, b) => b.score - a.score);

  return ranked;
}

// ==========================================================
// FEEDBACK BUCKET
// ==========================================================
function getFeedbackBucket(sentiment, intents) {
  const top = intents.filter(i => i.score >= 0.35).slice(0, 3);
  const names = top.map(t => t.label);

  if (names.includes("complaint") || sentiment === "negative") return "complaint";
  if (names.includes("bug_report")) return "bug_report";
  if (names.includes("refund") || names.includes("cancel_order"))
    return "refund/cancellation";
  if (names.includes("feature_request")) return "suggestion";
  if (names.includes("praise") || sentiment === "positive") return "praise";
  if (names.includes("question")) return "question";

  return "other";
}

// ==========================================================
// MAIN ANALYSIS PIPELINE
// ==========================================================
async function analyze(text) {
  const sentiment = await getSentiment(text);
  const intents = await getIntent(text);
  const bucket = getFeedbackBucket(sentiment.label, intents);

  return {
    input: text,
    sentiment,
    intents_ranked: intents.slice(0, 5),
    feedback_bucket: bucket
  };
}

// ==========================================================
// CLI INPUT LOOP
// ==========================================================
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const rl = readline.createInterface({ input, output });

console.log("\n💬 Type a message to analyze (Ctrl+C to exit):\n");

while (true) {
  const text = await rl.question("> ");
  if (!text.trim()) continue;

  const result = await analyze(text.trim());
  console.log("\n📌 RESULT:\n", JSON.stringify(result, null, 2), "\n");
}
