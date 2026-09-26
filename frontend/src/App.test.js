import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import App from './App.vue'

vi.mock('./services/searchService', () => ({
  isDemoMode: false,
  askQuestion: vi.fn(() => Promise.resolve({
    answer: '**Attention** uses several learned views of the input.\n\n* One head tracks syntax\n* Another tracks meaning',
    citations: [{
      id: 'attention',
      title: 'Attention notes',
      source: 'attention.md',
      excerpt: 'Several heads learn different relationships.',
      score: 0.91,
    }],
  })),
}))

describe('NextSearch question interface', () => {
  it('starts with a clear empty state and disabled submit', () => {
    const wrapper = mount(App)

    expect(wrapper.text()).toContain('Your research desk is ready.')
    expect(wrapper.get('button[type="submit"]').attributes('disabled')).toBeDefined()
  })

  it('renders an answer and its citations after asking a question', async () => {
    const wrapper = mount(App)
    await wrapper.get('input').setValue('How does attention work?')
    await wrapper.get('form').trigger('submit')
    await vi.waitFor(() => expect(wrapper.text()).toContain('Attention uses several learned views'))

    expect(wrapper.get('.answer-text strong').text()).toBe('Attention')
    expect(wrapper.findAll('.answer-text li')).toHaveLength(2)
    expect(wrapper.text()).toContain('Attention notes')
    expect(wrapper.text()).toContain('attention.md')
  })

  it('shows an actionable error state when the adapter fails', async () => {
    const { askQuestion } = await import('./services/searchService')
    askQuestion.mockRejectedValueOnce(new Error('Service unavailable'))
    const wrapper = mount(App)

    await wrapper.get('input').setValue('Will this fail?')
    await wrapper.get('form').trigger('submit')
    await vi.waitFor(() => expect(wrapper.get('[role="alert"]').text()).toContain('Service unavailable'))
    expect(wrapper.text()).toContain('Try another question')
  })
})
