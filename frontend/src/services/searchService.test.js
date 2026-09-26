import { describe, expect, it, vi } from 'vitest'
import { askQuestion } from './searchService'

describe('searchService', () => {
  it('posts a question and returns grounded citation data', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: 'Attention uses multiple heads.',
        citations: [{
          id: 'chunk-1',
          title: 'Attention notes',
          source: 'attention.md',
          excerpt: 'Several learned heads.',
          score: 0.92,
        }],
      }),
    })

    const result = await askQuestion('How does attention work?', { fetchImpl })

    expect(fetchImpl).toHaveBeenCalledWith('/api/ask', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'How does attention work?', top_k: 5 }),
    }))
    expect(result.citations[0]).toMatchObject({ id: 'chunk-1', source: 'attention.md' })
  })

  it('surfaces configured backend failures without demo fallback', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ ok: false, status: 503 })

    await expect(askQuestion('Question?', { fetchImpl })).rejects.toThrow('Search service returned 503.')
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('rejects malformed backend responses', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ citations: [] }) })

    await expect(askQuestion('Question?', { fetchImpl })).rejects.toThrow('invalid answer')
  })
})
