import { describe, expect, it } from 'vitest'
import { renderAnswerMarkdown } from './renderMarkdown'

describe('renderAnswerMarkdown', () => {
  it('renders emphasis and Markdown lists', () => {
    const html = renderAnswerMarkdown('**Bold** text\n\n* first\n* second')
    const container = document.createElement('div')
    container.innerHTML = html

    expect(container.querySelector('strong')?.textContent).toBe('Bold')
    expect(container.querySelectorAll('ul li')).toHaveLength(2)
    expect(container.textContent).toContain('first')
  })

  it('keeps bracketed citation markers as text', () => {
    const html = renderAnswerMarkdown('See the sidecar pattern [1].')
    const container = document.createElement('div')
    container.innerHTML = html

    expect(container.textContent).toContain('[1]')
    expect(container.querySelector('a')).toBeNull()
  })

  it('does not create active elements from raw HTML', () => {
    const html = renderAnswerMarkdown('<script>alert(1)</script><img src=x onerror=alert(1)>')
    const container = document.createElement('div')
    container.innerHTML = html

    expect(container.querySelector('script, img')).toBeNull()
    expect(container.textContent).toContain('<script>')
  })

  it('blocks dangerous link schemes', () => {
    const html = renderAnswerMarkdown('[click](javascript:alert(1))')
    const container = document.createElement('div')
    container.innerHTML = html

    const link = container.querySelector('a')
    expect(link?.getAttribute('href') ?? '').not.toMatch(/^javascript:/i)
  })

  it.each([null, undefined, '', 42])('returns empty HTML for invalid input %s', (input) => {
    expect(renderAnswerMarkdown(input)).toBe('')
  })
})
