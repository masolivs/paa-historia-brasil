<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { askQuestion } from './services/api';
import type { AnswerResponse } from './types/api';

type Citations = AnswerResponse['citations'];
type Timings = AnswerResponse['timings'];

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  model?: string;
  timings?: Timings;
  citations?: Citations;
  isError?: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

const STORAGE_CHATS_KEY = 'paa_rag_chats_v1';
const STORAGE_ACTIVE_KEY = 'paa_rag_active_chat_v1';
const STORAGE_TOPK_KEY = 'paa_rag_topk_v1';

const exampleQuestions = [
  'O que é astronomia e quais assuntos ela costuma estudar?',
  'Quais idiomas são usados principalmente na América Latina?',
  'Quem foi Albert Einstein e quais foram suas principais contribuições?',
];

function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function loadSavedChats(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_CHATS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function loadSavedTopK(): number {
  const raw = Number(localStorage.getItem(STORAGE_TOPK_KEY));
  return [3, 4, 5, 8].includes(raw) ? raw : 4;
}

const chats = ref<ChatSession[]>(loadSavedChats());
const activeChatId = ref<string | null>(localStorage.getItem(STORAGE_ACTIVE_KEY));
const topK = ref<number>(loadSavedTopK());

const question = ref('');
const loading = ref(false);
const sidebarOpen = ref(true);
const expandedSources = ref<Record<string, boolean>>({});
const confirmingClear = ref(false);

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const scrollContainerRef = ref<HTMLDivElement | null>(null);

if (activeChatId.value && !chats.value.some((c) => c.id === activeChatId.value)) {
  activeChatId.value = chats.value[0]?.id ?? null;
}

const sortedChats = computed(() =>
  [...chats.value].sort((a, b) => b.updatedAt - a.updatedAt)
);

const activeChat = computed<ChatSession | null>(
  () => chats.value.find((c) => c.id === activeChatId.value) ?? null
);

watch(
  chats,
  (val) => {
    try {
      localStorage.setItem(STORAGE_CHATS_KEY, JSON.stringify(val));
    } catch {
      // Ignora erro caso o limite do localStorage seja atingido
    }
  },
  { deep: true }
);

watch(activeChatId, (val) => {
  if (val) {
    localStorage.setItem(STORAGE_ACTIVE_KEY, val);
  } else {
    localStorage.removeItem(STORAGE_ACTIVE_KEY);
  }
});

watch(topK, (val) => {
  localStorage.setItem(STORAGE_TOPK_KEY, String(val));
});

onMounted(() => {
  if (window.innerWidth < 768) {
    sidebarOpen.value = false;
  }
});

function scrollToBottom() {
  nextTick(() => {
    const el = scrollContainerRef.value;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  });
}

function autoResizeTextarea() {
  const el = textareaRef.value;
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}

function startNewChat() {
  if (loading.value) return;
  activeChatId.value = null;
  question.value = '';
  nextTick(() => {
    autoResizeTextarea();
    textareaRef.value?.focus();
  });
  if (window.innerWidth < 768) {
    sidebarOpen.value = false;
  }
}

function selectChat(id: string) {
  if (loading.value) return;
  activeChatId.value = id;
  if (window.innerWidth < 768) {
    sidebarOpen.value = false;
  }
  scrollToBottom();
}

function deleteChat(id: string) {
  if (loading.value) return;
  chats.value = chats.value.filter((c) => c.id !== id);
  if (activeChatId.value === id) {
    activeChatId.value = sortedChats.value[0]?.id ?? null;
  }
}

function clearAllChats() {
  if (loading.value) return;
  if (!confirmingClear.value) {
    confirmingClear.value = true;
    setTimeout(() => {
      confirmingClear.value = false;
    }, 3000);
    return;
  }
  chats.value = [];
  activeChatId.value = null;
  expandedSources.value = {};
  confirmingClear.value = false;
}

function toggleSources(messageId: string) {
  expandedSources.value[messageId] = !expandedSources.value[messageId];
}

function formatChatTitle(text: string): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > 44 ? `${clean.slice(0, 44)}...` : clean;
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleDateString('pt-BR', {
    day: '2-digit',
    month: '2-digit',
  });
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSubmit();
  }
}

async function handleSubmit() {
  const trimmed = question.value.trim();
  if (trimmed.length < 3 || loading.value) return;

  const now = Date.now();
  let currentChat = activeChat.value;

  if (!currentChat) {
    const created: ChatSession = {
      id: generateId(),
      title: formatChatTitle(trimmed),
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    chats.value.unshift(created);
    activeChatId.value = created.id;
    currentChat = created;
  }

  currentChat.messages.push({
    id: generateId(),
    role: 'user',
    content: trimmed,
    createdAt: now,
  });
  currentChat.updatedAt = now;

  question.value = '';
  nextTick(() => autoResizeTextarea());
  scrollToBottom();

  loading.value = true;

  try {
    const response = await askQuestion(trimmed, topK.value);
    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: response.answer,
      createdAt: Date.now(),
      model: response.model,
      timings: response.timings,
      citations: response.citations,
    };
    currentChat.messages.push(assistantMsg);
    currentChat.updatedAt = Date.now();
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Erro desconhecido ao consultar o servidor.';
    currentChat.messages.push({
      id: generateId(),
      role: 'assistant',
      content: message,
      createdAt: Date.now(),
      isError: true,
    });
    currentChat.updatedAt = Date.now();
  } finally {
    loading.value = false;
    scrollToBottom();
  }
}

function selectExample(example: string) {
  question.value = example;
  handleSubmit();
}
</script>

<template>
  <div class="h-screen flex bg-slate-50 text-slate-800 overflow-hidden">
    <!-- Fundo escuro no mobile quando a barra lateral está aberta -->
    <div
      v-if="sidebarOpen"
      @click="sidebarOpen = false"
      class="fixed inset-0 bg-slate-900/30 z-20 md:hidden"
    />

    <!-- Barra lateral (Sidebar) -->
    <aside
      :class="[
        'fixed md:static inset-y-0 left-0 z-30 flex flex-col bg-slate-100 border-r border-slate-200 transition-all duration-200 shrink-0',
        sidebarOpen
          ? 'w-72 translate-x-0'
          : 'w-72 -translate-x-full md:w-0 md:translate-x-0 md:border-r-0 md:overflow-hidden',
      ]"
    >
      <!-- Topo da barra lateral -->
      <div class="h-14 px-3 flex items-center justify-between shrink-0">
        <button
          type="button"
          @click="sidebarOpen = false"
          title="Recolher barra lateral"
          class="p-2 rounded-lg text-slate-600 hover:bg-slate-200/80 transition"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        <button
          type="button"
          @click="startNewChat"
          :disabled="loading"
          class="flex items-center gap-2 rounded-full bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 text-white text-xs font-semibold px-3.5 py-2 shadow-sm transition"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          Nova conversa
        </button>
      </div>

      <!-- Lista de conversas salvas -->
      <div class="flex-1 overflow-y-auto px-2.5 py-2 space-y-1">
        <p class="px-2.5 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Recentes
        </p>

        <div
          v-if="sortedChats.length === 0"
          class="px-2.5 py-6 text-xs text-slate-500 text-center"
        >
          Nenhuma conversa salva ainda.
        </div>

        <div
          v-for="chat in sortedChats"
          :key="chat.id"
          @click="selectChat(chat.id)"
          :class="[
            'group flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl cursor-pointer text-sm transition',
            chat.id === activeChatId
              ? 'bg-emerald-100/80 text-emerald-950 font-medium'
              : 'text-slate-700 hover:bg-slate-200/70',
          ]"
        >
          <div class="flex items-center gap-2.5 min-w-0">
            <svg
              class="w-4 h-4 shrink-0 text-slate-400"
              :class="chat.id === activeChatId ? 'text-emerald-700' : ''"
              fill="none"
              stroke="currentColor"
              stroke-width="1.8"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M8 10h8M8 14h5m-7 6 3-3h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v13Z"
              />
            </svg>
            <div class="min-w-0">
              <p class="truncate text-xs leading-snug">{{ chat.title }}</p>
              <p class="text-[10px] text-slate-400">{{ formatTimestamp(chat.updatedAt) }}</p>
            </div>
          </div>

          <button
            type="button"
            @click.stop="deleteChat(chat.id)"
            title="Excluir conversa"
            class="opacity-0 group-hover:opacity-100 p-1 rounded-md text-slate-400 hover:text-red-600 hover:bg-white/70 transition shrink-0"
          >
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M19 7l-.867 12.142A2 2 0 0 1 16.138 21H7.862a2 2 0 0 1-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v3M4 7h16"
              />
            </svg>
          </button>
        </div>
      </div>

      <!-- Rodapé da barra lateral -->
      <div class="p-3 border-t border-slate-200 flex items-center justify-between gap-2 text-[11px] text-slate-500">
        <span>Salvo localmente</span>
        <button
          v-if="chats.length > 0"
          type="button"
          @click="clearAllChats"
          class="font-medium transition"
          :class="confirmingClear ? 'text-red-600' : 'text-slate-500 hover:text-red-600'"
        >
          {{ confirmingClear ? 'Confirmar limpeza?' : 'Limpar tudo' }}
        </button>
      </div>
    </aside>

    <!-- Área principal do chat -->
    <div class="flex-1 flex flex-col min-w-0 h-full bg-white">
      <!-- Barra superior minimalista -->
      <div class="h-14 px-4 flex items-center justify-between border-b border-slate-100 shrink-0">
        <div class="flex items-center gap-2 min-w-0">
          <button
            v-if="!sidebarOpen"
            type="button"
            @click="sidebarOpen = true"
            title="Abrir barra lateral"
            class="p-2 rounded-lg text-slate-600 hover:bg-slate-100 transition"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <span
            v-if="activeChat"
            class="text-sm font-medium text-slate-700 truncate"
          >
            {{ activeChat.title }}
          </span>
        </div>

        <button
          v-if="!sidebarOpen"
          type="button"
          @click="startNewChat"
          :disabled="loading"
          title="Nova conversa"
          class="flex items-center gap-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-50 px-3 py-1.5 rounded-full transition"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          Novo chat
        </button>
      </div>

      <!-- Histórico de mensagens ou tela inicial limpa -->
      <div ref="scrollContainerRef" class="flex-1 overflow-y-auto">
        <!-- Tela inicial (quando não há mensagens) -->
        <div
          v-if="!activeChat || activeChat.messages.length === 0"
          class="h-full max-w-3xl mx-auto px-4 flex flex-col items-center justify-center text-center py-10"
        >
          <h1 class="text-2xl sm:text-3xl font-semibold text-slate-800 tracking-tight">
            Como posso ajudar hoje?
          </h1>

          <div class="mt-8 flex flex-wrap justify-center gap-2 max-w-2xl">
            <button
              v-for="ex in exampleQuestions"
              :key="ex"
              type="button"
              @click="selectExample(ex)"
              :disabled="loading"
              class="text-xs sm:text-sm text-left bg-slate-50 hover:bg-emerald-50 text-slate-700 hover:text-emerald-900 border border-slate-200 hover:border-emerald-200 px-3.5 py-2 rounded-2xl transition"
            >
              {{ ex }}
            </button>
          </div>
        </div>

        <!-- Lista de mensagens do chat ativo -->
        <div v-else class="max-w-3xl mx-auto px-4 py-6 space-y-6">
          <div
            v-for="msg in activeChat.messages"
            :key="msg.id"
            class="flex"
            :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
          >
            <!-- Mensagem do usuário -->
            <div
              v-if="msg.role === 'user'"
              class="max-w-[85%] sm:max-w-[75%] rounded-2xl rounded-tr-sm bg-emerald-50 border border-emerald-200/70 px-4 py-2.5 text-sm text-slate-800 leading-relaxed whitespace-pre-line"
            >
              {{ msg.content }}
            </div>

            <!-- Mensagem do assistente -->
            <div v-else class="w-full min-w-0 space-y-3">
              <!-- Cabeçalho discreto da resposta (modelo) -->
              <div v-if="msg.model" class="flex items-center gap-2">
                <span class="text-[11px] font-mono bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-0.5 rounded-full">
                  Modelo: {{ msg.model }}
                </span>
              </div>

              <!-- Texto da resposta ou erro -->
              <div
                v-if="msg.isError"
                class="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                {{ msg.content }}
              </div>
              <p
                v-else
                class="text-sm sm:text-[15px] text-slate-800 leading-relaxed whitespace-pre-line"
              >
                {{ msg.content }}
              </p>

              <!-- Métricas de desempenho -->
              <div
                v-if="msg.timings"
                class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-xs"
              >
                <div class="bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-2">
                  <p class="text-[11px] text-slate-500">Embedding</p>
                  <p class="font-semibold text-slate-800">{{ msg.timings.embedding_ms }} ms</p>
                </div>
                <div class="bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-2">
                  <p class="text-[11px] text-slate-500">Busca Vetorial</p>
                  <p class="font-semibold text-slate-800">{{ msg.timings.search_ms }} ms</p>
                </div>
                <div class="bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-2">
                  <p class="text-[11px] text-slate-500">Inferência LLM</p>
                  <p class="font-semibold text-slate-800">{{ msg.timings.llm_ms }} ms</p>
                </div>
                <div class="bg-emerald-50 border border-emerald-100 rounded-lg px-2.5 py-2">
                  <p class="text-[11px] text-emerald-700">Tempo Total</p>
                  <p class="font-semibold text-emerald-900">{{ msg.timings.total_ms }} ms</p>
                </div>
              </div>

              <!-- Documentos citados (recolhidos por padrão) -->
              <div v-if="msg.citations && msg.citations.length > 0" class="pt-0.5">
                <button
                  type="button"
                  @click="toggleSources(msg.id)"
                  class="inline-flex items-center gap-2 text-xs font-medium text-slate-600 hover:text-emerald-800 bg-slate-50 hover:bg-emerald-50 border border-slate-200 hover:border-emerald-200 px-3 py-1.5 rounded-full transition"
                >
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                    />
                  </svg>
                  <span>
                    {{ expandedSources[msg.id] ? 'Ocultar fontes' : 'Consultar fontes' }}
                    ({{ msg.citations.length }})
                  </span>
                  <svg
                    class="w-3.5 h-3.5 transition-transform duration-150"
                    :class="expandedSources[msg.id] ? 'rotate-180' : ''"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                <div
                  v-if="expandedSources[msg.id]"
                  class="mt-3 space-y-2.5"
                >
                  <div
                    v-for="(citation, idx) in msg.citations"
                    :key="idx"
                    class="bg-slate-50/70 rounded-xl border border-slate-200 p-3.5"
                  >
                    <div class="flex items-center justify-between gap-2">
                      <h4 class="font-semibold text-xs sm:text-sm text-emerald-900">
                        [Fonte {{ idx + 1 }}] {{ citation.document }}
                      </h4>
                      <span class="text-[11px] font-mono text-slate-500 shrink-0">
                        Score: {{ citation.score.toFixed(4) }}
                      </span>
                    </div>
                    <p class="text-[11px] text-slate-500 mt-0.5">{{ citation.source }}</p>
                    <a
                      v-if="citation.source_url"
                      :href="citation.source_url"
                      target="_blank"
                      rel="noopener noreferrer"
                      class="inline-block mt-1 text-xs text-emerald-700 hover:underline break-all"
                    >
                      Consultar documento original
                    </a>
                    <p class="text-xs sm:text-sm text-slate-700 mt-2 leading-relaxed">
                      "{{ citation.snippet }}"
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Indicador de carregamento -->
          <div v-if="loading" class="pt-1 text-sm text-slate-500 flex items-center gap-2">
            <span class="inline-flex gap-1">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-bounce" />
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-bounce [animation-delay:120ms]" />
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-bounce [animation-delay:240ms]" />
            </span>
            Buscando documentos e gerando resposta...
          </div>
        </div>
      </div>

      <!-- Caixa de pergunta fixa na parte inferior -->
      <div class="px-4 pb-4 pt-2 bg-white shrink-0">
        <form
          @submit.prevent="handleSubmit"
          class="max-w-3xl mx-auto rounded-2xl border border-slate-300 bg-slate-50 focus-within:bg-white focus-within:border-emerald-600 focus-within:ring-2 focus-within:ring-emerald-100 shadow-sm transition px-4 py-2.5"
        >
          <textarea
            ref="textareaRef"
            v-model="question"
            rows="1"
            @input="autoResizeTextarea"
            @keydown="handleKeydown"
            placeholder="Digite sua pergunta..."
            :disabled="loading"
            class="w-full resize-none bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none max-h-40 py-1"
          />

          <div class="flex items-center justify-between gap-2 pt-1">
            <label class="text-xs text-slate-500 flex items-center gap-1.5">
              <span>Fontes (top-k):</span>
              <select
                v-model.number="topK"
                :disabled="loading"
                class="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs text-slate-700 focus:border-emerald-600 focus:outline-none"
              >
                <option :value="3">3</option>
                <option :value="4">4</option>
                <option :value="5">5</option>
                <option :value="8">8</option>
              </select>
            </label>

            <button
              type="submit"
              :disabled="loading || question.trim().length < 3"
              title="Enviar pergunta"
              class="inline-flex items-center justify-center w-9 h-9 rounded-full bg-emerald-700 text-white hover:bg-emerald-800 disabled:opacity-40 disabled:hover:bg-emerald-700 transition"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14m-6-6 6 6-6 6" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>