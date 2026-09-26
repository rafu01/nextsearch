<script setup>
import { computed, ref } from 'vue'
import { askQuestion, isDemoMode } from './services/searchService'

const question = ref('')
const state = ref('empty')
const result = ref(null)
const errorMessage = ref('')
const controller = ref(null)

const canSubmit = computed(() => question.value.trim().length > 0 && state.value !== 'loading')

async function submitQuestion() {
  const query = question.value.trim()
  if (!query || state.value === 'loading') return

  state.value = 'loading'
  result.value = null
  errorMessage.value = ''
  controller.value = new AbortController()

  try {
    result.value = await askQuestion(query, { signal: controller.value.signal })
    state.value = 'success'
  } catch (error) {
    if (error.name === 'AbortError') return
    errorMessage.value = error.message || 'Something went wrong while searching.'
    state.value = 'error'
  } finally {
    controller.value = null
  }
}

function resetSearch() {
  controller.value?.abort()
  question.value = ''
  result.value = null
  errorMessage.value = ''
  state.value = 'empty'
}
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="NextSearch home">
        <span class="brand-mark">N</span>
        <span>nextsearch</span>
      </a>
      <div class="status-pill"><span class="status-dot" /> {{ isDemoMode ? 'Demo preview' : 'Connected search' }}</div>
    </header>

    <section class="hero" aria-labelledby="page-title">
      <p class="eyebrow">Research, without the tab sprawl</p>
      <h1 id="page-title">Ask your notes<br /><em>better questions.</em></h1>
      <p class="intro">Search across your Obsidian notes and textbooks. Every answer stays grounded in the sources you indexed.</p>

      <form class="search-card" @submit.prevent="submitQuestion">
        <label for="question">What would you like to understand?</label>
        <div class="search-row">
          <input
            id="question"
            v-model="question"
            type="text"
            autocomplete="off"
            placeholder="e.g. How does backpropagation work?"
            :disabled="state === 'loading'"
          />
          <button type="submit" :disabled="!canSubmit">
            <span v-if="state === 'loading'" class="spinner" aria-hidden="true" />
            <span v-else>Ask <span aria-hidden="true">↗</span></span>
          </button>
        </div>
        <div class="search-hint"><span>⌘</span> Enter to search <span class="hint-divider" /> Answers cite your indexed sources</div>
      </form>
    </section>

    <section class="content" aria-live="polite">
      <div v-if="state === 'empty'" class="empty-state">
        <div class="empty-icon">✦</div>
        <div>
          <h2>Your research desk is ready.</h2>
          <p>Ask a question above to find a concise answer with the notes that support it.</p>
        </div>
      </div>

      <div v-else-if="state === 'loading'" class="loading-state">
        <div class="loading-bar" />
        <p>Searching your knowledge base<span class="ellipsis">...</span></p>
        <span>Ranking passages and preparing citations</span>
      </div>

      <div v-else-if="state === 'error'" class="error-state" role="alert">
        <div class="error-icon">!</div>
        <div>
          <h2>We couldn't find an answer.</h2>
          <p>{{ errorMessage }}</p>
          <button class="text-button" type="button" @click="state = 'empty'">Try another question →</button>
        </div>
      </div>

      <article v-else class="answer-card">
        <div class="answer-heading">
          <div>
            <p class="eyebrow">Grounded answer</p>
            <h2>{{ question }}</h2>
          </div>
          <button class="new-search" type="button" @click="resetSearch">New search <span>＋</span></button>
        </div>
        <div class="answer-body">
          <div class="answer-mark">“</div>
          <p>{{ result.answer }}</p>
        </div>
        <div class="sources-heading"><span>Sources</span><span class="source-count">{{ result.citations.length }} cited</span></div>
        <ul v-if="result.citations.length" class="sources-list">
          <li v-for="citation in result.citations" :key="citation.id" class="source-item">
            <div class="source-index">{{ String(result.citations.indexOf(citation) + 1).padStart(2, '0') }}</div>
            <div class="source-copy">
              <strong>{{ citation.title }}</strong>
              <span>{{ citation.source }}</span>
              <p>{{ citation.excerpt }}</p>
            </div>
            <div class="source-score">{{ Math.round(citation.score * 100) }}%</div>
          </li>
        </ul>
        <p v-else class="no-sources">No source passages were returned for this answer.</p>
      </article>
    </section>

    <footer><span>nextsearch</span><span>Private by default · Built for deep work</span></footer>
  </main>
</template>
