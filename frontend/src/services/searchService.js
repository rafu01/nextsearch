/** Adapter from the Vue UI to the NextSearch HTTP API. */
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
export const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true'

const ASK_ENDPOINT = `${API_BASE_URL}/api/ask`

export const demoResponse = {
  answer:
    'Multi-head attention lets a transformer attend to several representation subspaces at once. Each head can focus on a different relationship, and the results are concatenated before the final projection.',
  citations: [
    {
      id: 'demo-attention',
      title: 'Attention Is All You Need',
      source: 'transformers/attention.md',
      excerpt: 'Multi-head attention allows the model to jointly attend to information from different representation subspaces.',
      score: 0.94,
    },
  ],
}

function normalizeResponse(payload) {
  if (!payload || typeof payload.answer !== 'string') {
    throw new Error('The search service returned an invalid answer.')
  }

  return {
    answer: payload.answer,
    citations: Array.isArray(payload.citations) ? payload.citations : [],
  }
}

export async function askQuestion(query, { signal, fetchImpl = fetch } = {}) {
  if (isDemoMode) {
    await new Promise((resolve) => setTimeout(resolve, 450))
    return demoResponse
  }

  const response = await fetchImpl(ASK_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 5 }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Search service returned ${response.status}.`)
  }

  return normalizeResponse(await response.json())
}

export async function fetchStats({ fetchImpl = fetch } = {}) {
  if (!API_BASE_URL) {
    return { chunks_indexed: 0, sources: 0 }
  }

  const response = await fetchImpl(`${API_BASE_URL}/api/stats`)
  if (!response.ok) throw new Error(`Stats service returned ${response.status}.`)
  return response.json()
}
