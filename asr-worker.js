// Spracherkennung direkt im Handy: Whisper (base) über transformers.js.
// Läuft komplett im Browser, nichts geht an einen Server. Das Modell (ca. 80 MB) wird einmal geladen und dann im Browser gespeichert.
// Die App gibt eine kurze Wortliste mit (Gerichte von heute, typische Lebensmittel). Damit schreibt das Modell
// z. B. „Schinken-Käse-Toast“ statt „Schinkenkehse Tors“.
import { AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
env.allowLocalModels = false;
env.useBrowserCache = true;
const MODEL = 'onnx-community/whisper-base';
let M = null, loading = null;
const progress = p => { if (p && (p.status === 'progress' || p.status === 'initiate' || p.status === 'done')) self.postMessage({ type: 'progress', file: p.file, loaded: p.loaded || 0, total: p.total || 0, status: p.status }); };
async function loadOnce() {
  const [processor, tokenizer, model] = await Promise.all([
    AutoProcessor.from_pretrained(MODEL),
    AutoTokenizer.from_pretrained(MODEL),
    WhisperForConditionalGeneration.from_pretrained(MODEL, { dtype: { encoder_model: 'q8', decoder_model_merged: 'q8' }, device: 'wasm', progress_callback: progress }),
  ]);
  return { processor, tokenizer, model };
}
function load() {
  if (M) return Promise.resolve(M);
  // Ein Wackler im Netz soll nicht gleich alles abbrechen: einmal neu versuchen
  if (!loading) loading = loadOnce().catch(() => new Promise(r => setTimeout(r, 1500)).then(loadOnce))
    .then(m => { M = m; self.postMessage({ type: 'ready' }); return m; })
    .catch(e => { loading = null; throw e; });
  return loading;
}
async function transcribe(audio, prompt) {
  const { processor, tokenizer, model } = await load();
  const inputs = await processor(audio);
  const tid = s => tokenizer.model.tokens_to_ids.get(s);
  let ids = [tid('<|startoftranscript|>'), tid('<|de|>'), tid('<|transcribe|>'), tid('<|notimestamps|>')];
  if (prompt) ids = [tid('<|startofprev|>'), ...tokenizer.encode(' ' + String(prompt).trim(), { add_special_tokens: false }).slice(-120), ...ids];
  const secs = audio.length / 16000;
  const out = await model.generate({ ...inputs, decoder_input_ids: ids, max_new_tokens: Math.min(160, Math.round(secs * 9) + 16) });
  const seq = (out.tolist ? out.tolist()[0] : Array.from(out[0])).map(Number).slice(ids.length);
  return tokenizer.decode(seq, { skip_special_tokens: true }).trim();
}
self.onmessage = async e => {
  const m = e.data || {};
  try {
    if (m.type === 'load') { await load(); return; }
    if (m.type === 'run') self.postMessage({ type: 'result', id: m.id, text: await transcribe(m.audio, m.prompt) });
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String((err && err.message) || err) }); }
};
