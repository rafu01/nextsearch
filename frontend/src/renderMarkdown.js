import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'

const markdown = new MarkdownIt({
  html: false,
  linkify: false,
  breaks: true,
  typographer: false,
})

const SANITIZE_OPTIONS = {
  ALLOWED_TAGS: [
    'a', 'blockquote', 'br', 'code', 'del', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'hr', 'li', 'ol', 'p', 'pre', 'strong', 'ul',
  ],
  ALLOWED_ATTR: ['href'],
  ALLOW_DATA_ATTR: false,
}

export function renderAnswerMarkdown(rawText) {
  if (typeof rawText !== 'string' || !rawText) return ''

  const html = markdown.render(rawText)
  // This sanitized string is bound with v-html; never bypass this step.
  return DOMPurify.sanitize(html, SANITIZE_OPTIONS)
}
