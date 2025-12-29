"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface Feature {
  feature_idx: number
  activation: number
  explanation: string
}

interface TokenExplanation {
  token: string
  features: Feature[]
}

interface OptimizedFeature {
  feature_idx: number
  max_act: number
  steering_strength: number
  reasoning: string
  attribute: string
  feature_description: string
}

interface SteeringResult {
  prompt: string
  steering_prompt: string
  baseline_text: string
  steered_text: string
  extracted_attributes: string[]
  attribute_reasoning: string
  optimized_features: OptimizedFeature[]
  overall_strategy: string
}

interface CandidateFeature {
  attribute: string
  feature_idx: number
  relevance_score: number
  feature_description: string
  max_act?: number
}

interface PreviewResult {
  prompt: string
  steering_prompt: string
  extracted_attributes: string[]
  attribute_reasoning: string
  candidate_features: CandidateFeature[]
}

interface GenerateResult {
  prompt: string
  full_text: string
  input_token_explanations: TokenExplanation[]
  output_token_explanations: TokenExplanation[]
  feature_max_activations: Record<string, number>
}

interface TunedFeature extends CandidateFeature {
  enabled: boolean
  strength: number
}

interface RandomFeature {
  feature_idx: number
  description: string
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  steeringPrompt?: string
  result?: {
    steering?: SteeringResult
    generate?: GenerateResult
  }
}

export function AutoSteerDemo() {
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [input, setInput] = React.useState("")
  const [loading, setLoading] = React.useState(false)
  const [selectedTokenIdx, setSelectedTokenIdx] = React.useState<number | null>(null)
  const [activeMessageId, setActiveMessageId] = React.useState<string | null>(null)

  const [tuningModalOpen, setTuningModalOpen] = React.useState(false)
  const [pendingPrompt, setPendingPrompt] = React.useState("")
  const [pendingSteeringPrompt, setPendingSteeringPrompt] = React.useState("")
  const [previewLoading, setPreviewLoading] = React.useState(false)
  const [tunedFeatures, setTunedFeatures] = React.useState<TunedFeature[]>([])
  const [extractedAttributes, setExtractedAttributes] = React.useState<string[]>([])
  const [randomFeatures, setRandomFeatures] = React.useState<RandomFeature[]>([])
  const [randomFeaturesLoading, setRandomFeaturesLoading] = React.useState(false)

  const chatEndRef = React.useRef<HTMLDivElement>(null)

  const activeMessage = messages.find((m) => m.id === activeMessageId)
  const activeResult = activeMessage?.result

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  React.useEffect(() => {
    scrollToBottom()
  }, [messages])

  const fetchPreview = async (prompt: string, steeringPrompt: string) => {
    if (!steeringPrompt.trim() || !prompt.trim()) {
      setTunedFeatures([])
      return
    }

    setPreviewLoading(true)
    try {
      const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || ""
      const response = await fetch(`${serverUrl}/preview_steering`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          steering_prompt: steeringPrompt,
          top_k_features: 5,
        }),
      })
      if (response.ok) {
        const data: PreviewResult = await response.json()
        setExtractedAttributes(data.extracted_attributes || [])
        setTunedFeatures(
          (data.candidate_features || []).map((f) => ({
            ...f,
            enabled: false,
            strength: 1.0,
          }))
        )
      }
    } catch {
      // Silently fail preview
    } finally {
      setPreviewLoading(false)
    }
  }

  const fetchRandomFeatures = async () => {
    setRandomFeaturesLoading(true)
    try {
      const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || ""
      const response = await fetch(`${serverUrl}/random_features?count=25`)
      if (response.ok) {
        const data = await response.json()
        setRandomFeatures(data.features || [])
      }
    } catch {
      // Silently fail
    } finally {
      setRandomFeaturesLoading(false)
    }
  }

  const handleOpenTuning = () => {
    if (!input.trim()) return
    setPendingPrompt(input)
    setPendingSteeringPrompt("")
    setTunedFeatures([])
    setExtractedAttributes([])
    setRandomFeatures([])
    setTuningModalOpen(true)
    fetchRandomFeatures()
  }

  const addRandomFeatureToTuned = (feature: RandomFeature) => {
    // Check if already added
    if (tunedFeatures.some((f) => f.feature_idx === feature.feature_idx)) {
      return
    }
    setTunedFeatures((prev) => [
      ...prev,
      {
        feature_idx: feature.feature_idx,
        feature_description: feature.description,
        attribute: "manual",
        relevance_score: 1.0,
        max_act: 30.0,
        enabled: true,
        strength: 1.0,
      },
    ])
  }

  const handleQuickGenerate = async () => {
    if (!input.trim()) return

    const messageId = Date.now().toString()
    const userMessage: ChatMessage = {
      id: messageId,
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setLoading(true)
    setSelectedTokenIdx(null)

    try {
      const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || ""

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000)

      const generateResponse = await fetch(`${serverUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: input,
          max_new_tokens: 100,
          top_k: 5,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!generateResponse.ok) {
        throw new Error(`Generate failed: ${generateResponse.status}`)
      }

      const generateData: GenerateResult = await generateResponse.json()

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: generateData.full_text,
        result: {
          generate: generateData,
        },
      }

      setMessages((prev) => [...prev, assistantMessage])
      setActiveMessageId(assistantMessage.id)
    } catch (err) {
      console.error("Generate error:", err)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Failed to generate"}`,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleSteeringChange = (value: string) => {
    setPendingSteeringPrompt(value)
    const timeoutId = setTimeout(() => {
      fetchPreview(pendingPrompt, value)
    }, 500)
    return () => clearTimeout(timeoutId)
  }

  const handleGenerate = async () => {
    if (!pendingPrompt.trim()) return

    const messageId = Date.now().toString()
    const userMessage: ChatMessage = {
      id: messageId,
      role: "user",
      content: pendingPrompt,
      steeringPrompt: pendingSteeringPrompt || undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setTuningModalOpen(false)
    setInput("")
    setLoading(true)
    setSelectedTokenIdx(null)

    try {
      const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || ""

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000)

      const enabledFeatures = tunedFeatures.filter((f) => f.enabled)
      const hasCustomSteering = enabledFeatures.length > 0

      const [steerResponse, generateResponse] = await Promise.all([
        hasCustomSteering
          ? fetch(`${serverUrl}/steer`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                prompt: pendingPrompt,
                steering_configs: enabledFeatures.map((f) => ({
                  steering_feature: f.feature_idx,
                  max_act: f.max_act || 30.0,
                  steering_strength: f.strength,
                })),
                max_new_tokens: 100,
              }),
              signal: controller.signal,
            })
          : pendingSteeringPrompt
            ? fetch(`${serverUrl}/auto_steer_from_prompt`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  prompt: pendingPrompt,
                  steering_prompt: pendingSteeringPrompt,
                  max_new_tokens: 100,
                  top_k_features: 3,
                }),
                signal: controller.signal,
              })
            : Promise.resolve(null),
        fetch(`${serverUrl}/generate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: pendingPrompt,
            max_new_tokens: 100,
            top_k: 5,
          }),
          signal: controller.signal,
        }),
      ])

      clearTimeout(timeoutId)

      if (!generateResponse.ok) {
        throw new Error(`Generate failed: ${generateResponse.status}`)
      }

      const generateData: GenerateResult = await generateResponse.json()

      let steerData: SteeringResult | null = null
      if (steerResponse && steerResponse.ok) {
        steerData = await steerResponse.json()
      }

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: steerData?.steered_text || generateData.full_text,
        result: {
          generate: generateData,
          steering: steerData || undefined,
        },
      }

      setMessages((prev) => [...prev, assistantMessage])
      setActiveMessageId(assistantMessage.id)
    } catch (err) {
      console.error("Generate error:", err)
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Failed to generate"}`,
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
      setPendingPrompt("")
      setPendingSteeringPrompt("")
      setTunedFeatures([])
    }
  }

  const allTokens = activeResult?.generate
    ? [
        ...(activeResult.generate.input_token_explanations || []).map((t) => ({
          ...t,
          type: "input" as const,
        })),
        ...(activeResult.generate.output_token_explanations || []).map((t) => ({
          ...t,
          type: "output" as const,
        })),
      ]
    : []

  const selectedToken = selectedTokenIdx !== null ? allTokens[selectedTokenIdx] : null
  const maxActivation = allTokens.length
    ? Math.max(...allTokens.flatMap((t) => (t.features || []).map((f) => f.activation)), 1)
    : 1

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Tuning Modal */}
      {tuningModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-8">
          <div className="flex h-full max-h-[800px] w-full max-w-4xl flex-col rounded-xl border border-border bg-background shadow-2xl">
            <div className="flex items-center justify-between border-b border-border px-6 py-4">
              <div>
                <h2 className="text-lg font-semibold">Tune Steering</h2>
                <p className="text-[13px] text-muted-foreground">Adjust feature strengths before generating</p>
              </div>
              <button
                onClick={() => setTuningModalOpen(false)}
                className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex flex-1 overflow-hidden">
              <div className="flex w-1/2 flex-col border-r border-border">
                <div className="flex-1 space-y-4 overflow-auto p-6">
                  <div>
                    <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      Prompt
                    </label>
                    <div className="rounded-lg border border-border bg-card p-4">
                      <p className="text-[14px]">{pendingPrompt}</p>
                    </div>
                  </div>

                  <div>
                    <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      Steering Instructions
                    </label>
                    <textarea
                      value={pendingSteeringPrompt}
                      onChange={(e) => handleSteeringChange(e.target.value)}
                      placeholder="Describe how you want to steer the output..."
                      rows={4}
                      className="w-full resize-none rounded-lg border border-border bg-card px-4 py-3 text-[14px] placeholder-muted-foreground/50 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>

                  {extractedAttributes.length > 0 && (
                    <div>
                      <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                        Detected Attributes
                      </label>
                      <div className="flex flex-wrap gap-2">
                        {extractedAttributes.map((attr, i) => (
                          <span key={i} className="rounded-full bg-primary/20 px-3 py-1 text-[12px] text-primary">
                            {attr}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div>
                    <label className="mb-2 block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      Browse Features
                    </label>
                    {randomFeaturesLoading ? (
                      <div className="flex items-center gap-2 py-3 text-[13px] text-muted-foreground">
                        <div className="h-3 w-3 animate-spin rounded-full border-2 border-muted border-t-primary" />
                        Loading features...
                      </div>
                    ) : (
                      <div className="max-h-[200px] space-y-2 overflow-auto rounded-lg border border-border bg-card p-2">
                        {randomFeatures.map((feature) => {
                          const isAdded = tunedFeatures.some((f) => f.feature_idx === feature.feature_idx)
                          return (
                            <div
                              key={feature.feature_idx}
                              className={cn(
                                "flex items-start gap-2 rounded-md p-2 text-[12px] transition-colors",
                                isAdded ? "bg-primary/10" : "hover:bg-muted"
                              )}
                            >
                              <button
                                onClick={() => addRandomFeatureToTuned(feature)}
                                disabled={isAdded}
                                className={cn(
                                  "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded border transition-colors",
                                  isAdded
                                    ? "border-primary bg-primary text-primary-foreground"
                                    : "border-muted-foreground/30 hover:border-primary hover:bg-primary/10"
                                )}
                              >
                                {isAdded ? (
                                  <svg className="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                                    <path
                                      fillRule="evenodd"
                                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                      clipRule="evenodd"
                                    />
                                  </svg>
                                ) : (
                                  <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                  </svg>
                                )}
                              </button>
                              <div className="flex-1 min-w-0">
                                <span className="font-mono text-[10px] text-primary">#{feature.feature_idx}</span>
                                <p className="text-muted-foreground leading-relaxed line-clamp-2">{feature.description}</p>
                              </div>
                            </div>
                          )
                        })}
                        {randomFeatures.length === 0 && (
                          <p className="py-2 text-center text-[12px] text-muted-foreground/70">No features loaded</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex w-1/2 flex-col">
                <div className="border-b border-border px-6 py-3">
                  <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                    Steering Features
                  </span>
                </div>
                <div className="flex-1 overflow-auto p-4">
                  {previewLoading && (
                    <div className="flex items-center gap-3 p-4 text-[13px] text-muted-foreground">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted border-t-primary" />
                      Finding relevant features...
                    </div>
                  )}

                  {!previewLoading && tunedFeatures.length === 0 && (
                    <div className="flex h-full items-center justify-center text-center">
                      <div>
                        <p className="text-[14px] text-muted-foreground/70">
                          {pendingSteeringPrompt ? "No features found" : "Enter steering instructions to find features"}
                        </p>
                        <p className="mt-1 text-[12px] text-muted-foreground/50">Or generate without steering</p>
                      </div>
                    </div>
                  )}

                  {!previewLoading && tunedFeatures.length > 0 && (
                    <div className="space-y-3">
                      {tunedFeatures.map((feature, idx) => (
                        <div
                          key={feature.feature_idx}
                          className={cn(
                            "rounded-lg border p-4 transition-all",
                            feature.enabled ? "border-primary/30 bg-primary/5" : "border-border bg-card opacity-50"
                          )}
                        >
                          <div className="mb-3 flex items-start justify-between">
                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => {
                                  const updated = [...tunedFeatures]
                                  updated[idx].enabled = !updated[idx].enabled
                                  setTunedFeatures(updated)
                                }}
                                className={cn(
                                  "flex h-5 w-5 items-center justify-center rounded border transition-colors",
                                  feature.enabled ? "border-primary bg-primary" : "border-muted-foreground/50"
                                )}
                              >
                                {feature.enabled && (
                                  <svg className="h-3 w-3 text-primary-foreground" fill="currentColor" viewBox="0 0 20 20">
                                    <path
                                      fillRule="evenodd"
                                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                      clipRule="evenodd"
                                    />
                                  </svg>
                                )}
                              </button>
                              <div>
                                <div className="flex items-center gap-2">
                                  <span className="font-mono text-[12px] text-primary">#{feature.feature_idx}</span>
                                  <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                                    {feature.attribute}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <span className="text-[11px] text-muted-foreground">
                              {(feature.relevance_score * 100).toFixed(0)}% match
                            </span>
                          </div>

                          <p className="mb-3 text-[12px] leading-relaxed text-muted-foreground">
                            {feature.feature_description}
                          </p>

                          {feature.enabled && (
                            <div className="flex items-center gap-4">
                              <span className="text-[11px] text-muted-foreground">Strength</span>
                              <input
                                type="range"
                                min="0"
                                max="3"
                                step="0.1"
                                value={feature.strength}
                                onChange={(e) => {
                                  const updated = [...tunedFeatures]
                                  updated[idx].strength = parseFloat(e.target.value)
                                  setTunedFeatures(updated)
                                }}
                                className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-muted accent-primary"
                              />
                              <span className="w-12 text-right font-mono text-[12px] text-primary">
                                {feature.strength.toFixed(1)}×
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between border-t border-border px-6 py-4">
              <button
                onClick={() => setTuningModalOpen(false)}
                className="text-[13px] text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleGenerate}
                className="rounded-lg bg-primary px-6 py-2.5 text-[13px] font-medium text-primary-foreground hover:bg-primary/90"
              >
                Generate
                {tunedFeatures.filter((f) => f.enabled).length > 0
                  ? ` with ${tunedFeatures.filter((f) => f.enabled).length} features`
                  : ""}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Left Panel: Chat */}
      <div className="flex w-[480px] flex-col border-r border-border">
        <div className="border-b border-border px-4 py-3">
          <span className="text-[12px] font-medium text-muted-foreground">CHAT</span>
        </div>

        <div className="flex-1 overflow-auto p-4">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center text-center">
              <div>
                <p className="text-[14px] text-muted-foreground/70">No messages yet</p>
                <p className="mt-1 text-[12px] text-muted-foreground/50">Type a prompt below to start</p>
              </div>
            </div>
          )}

          <div className="space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "rounded-lg p-3",
                  msg.role === "user" ? "bg-muted" : "border border-border bg-card"
                )}
              >
                <div className="mb-1 flex items-center gap-2">
                  <span className="text-[10px] font-medium uppercase text-muted-foreground">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </span>
                  {msg.steeringPrompt && (
                    <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[9px] text-primary">steered</span>
                  )}
                </div>
                <p className="text-[13px] leading-relaxed">{msg.content}</p>
                {msg.result && (
                  <button
                    onClick={() => {
                      setActiveMessageId(msg.id)
                      setSelectedTokenIdx(null)
                    }}
                    className={cn(
                      "mt-2 text-[11px]",
                      activeMessageId === msg.id ? "text-primary" : "text-muted-foreground hover:text-primary"
                    )}
                  >
                    {activeMessageId === msg.id ? "Viewing →" : "View analysis →"}
                  </button>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 p-3 text-[13px] text-muted-foreground">
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-muted border-t-primary" />
                Generating...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        <div className="border-t border-border p-4">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && input.trim() && !loading) {
                  e.preventDefault()
                  handleQuickGenerate()
                }
              }}
              placeholder="Enter your prompt..."
              rows={2}
              className="flex-1 resize-none rounded-lg border border-border bg-card px-3 py-2 text-[13px] placeholder-muted-foreground/50 focus:border-primary focus:outline-none"
            />
            <div className="flex flex-col gap-2">
              <button
                onClick={handleQuickGenerate}
                disabled={!input.trim() || loading}
                className="rounded-lg border border-border bg-card px-4 py-2 text-[12px] font-medium text-foreground hover:bg-muted disabled:opacity-50"
              >
                Send
              </button>
              <button
                onClick={handleOpenTuning}
                disabled={!input.trim() || loading}
                className="rounded-lg bg-primary px-4 py-2 text-[12px] font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                Tune
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel: Dashboard */}
      <div className="flex flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-border px-6 py-3">
          <div className="flex items-center gap-4">
            {activeResult ? (
              <>
                <span className="text-[12px] text-muted-foreground">{allTokens.length} tokens</span>
                {activeResult.steering?.optimized_features && activeResult.steering.optimized_features.length > 0 && (
                  <span className="rounded bg-primary/20 px-2 py-0.5 text-[11px] font-medium text-primary">
                    {activeResult.steering.optimized_features.length} features steered
                  </span>
                )}
              </>
            ) : (
              <span className="text-[12px] text-muted-foreground/70">Select a message to view analysis</span>
            )}
          </div>
        </div>

        {activeResult ? (
          <div className="flex flex-1 overflow-hidden">
            {/* Token Heatmap */}
            <div className="flex w-1/2 flex-col border-r border-border">
              <div className="border-b border-border px-4 py-2">
                <span className="text-[11px] font-medium text-muted-foreground">TOKEN ACTIVATIONS</span>
              </div>
              <div className="flex-1 overflow-auto p-4">
                <div className="flex flex-wrap gap-1">
                  {allTokens.map((token, idx) => {
                    const topAct = token.features?.[0]?.activation || 0
                    const intensity = maxActivation > 0 ? topAct / maxActivation : 0
                    const isSelected = selectedTokenIdx === idx
                    const opacityClass =
                      intensity < 0.2
                        ? "bg-primary/10"
                        : intensity < 0.4
                          ? "bg-primary/20"
                          : intensity < 0.6
                            ? "bg-primary/30"
                            : intensity < 0.8
                              ? "bg-primary/40"
                              : "bg-primary/50"
                    return (
                      <button
                        key={idx}
                        onClick={() => setSelectedTokenIdx(idx)}
                        className={cn(
                          "relative rounded px-1.5 py-1 font-mono text-[12px] transition-all",
                          opacityClass,
                          isSelected
                            ? "ring-2 ring-primary ring-offset-1 ring-offset-background"
                            : "hover:ring-1 hover:ring-primary/50",
                          token.type === "output" ? "border-b-2 border-primary" : ""
                        )}
                      >
                        {token.token.replace(/\s/g, "·")}
                      </button>
                    )
                  })}
                </div>
              </div>
              <div className="border-t border-border px-4 py-2">
                <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-4 rounded-sm bg-primary/20" />
                    <span>Low</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-4 rounded-sm bg-primary/60" />
                    <span>High</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 w-4 border-b-2 border-primary" />
                    <span>Output</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panels */}
            <div className="flex w-1/2 flex-col">
              {/* Token Features */}
              <div className="flex flex-1 flex-col overflow-hidden border-b border-border">
                <div className="border-b border-border px-4 py-2">
                  <span className="text-[11px] font-medium text-muted-foreground">
                    {selectedToken ? `FEATURES FOR "${selectedToken.token}"` : "SELECT A TOKEN"}
                  </span>
                </div>
                <div className="flex-1 overflow-auto p-4">
                  {selectedToken && selectedToken.features && selectedToken.features.length > 0 ? (
                    <div className="space-y-2">
                      {selectedToken.features.map((feature, idx) => {
                        const pct = (feature.activation / maxActivation) * 100
                        return (
                          <div key={idx} className="rounded-lg border border-border bg-card p-3">
                            <div className="mb-2 flex items-center justify-between">
                              <span className="font-mono text-[11px] text-primary">#{feature.feature_idx}</span>
                              <span className="font-mono text-[11px] text-muted-foreground">
                                {feature.activation.toFixed(2)}
                              </span>
                            </div>
                            <p className="mb-2 text-[12px] leading-relaxed text-muted-foreground">
                              {feature.explanation}
                            </p>
                            <div className="h-1.5 w-full rounded-full bg-muted">
                              <div
                                className="h-full rounded-full bg-primary"
                                style={{ width: `${Math.min(pct, 100)}%` }}
                              />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="flex h-full items-center justify-center text-[13px] text-muted-foreground/70">
                      {selectedToken ? "No features" : "Click a token to inspect"}
                    </div>
                  )}
                </div>
              </div>

              {/* Steered Features */}
              <div className="flex flex-1 flex-col overflow-hidden">
                <div className="border-b border-border px-4 py-2">
                  <span className="text-[11px] font-medium text-muted-foreground">
                    {activeResult.steering ? "APPLIED STEERING" : "OUTPUT PREVIEW"}
                  </span>
                </div>
                <div className="flex-1 overflow-auto p-4">
                  {activeResult.steering?.optimized_features &&
                  activeResult.steering.optimized_features.length > 0 ? (
                    <div className="space-y-3">
                      {activeResult.steering.overall_strategy && (
                        <div className="rounded-lg border border-border bg-card p-3">
                          <p className="text-[11px] text-muted-foreground">
                            {activeResult.steering.overall_strategy}
                          </p>
                        </div>
                      )}
                      {activeResult.steering.optimized_features.map((feature, idx) => (
                        <div key={idx} className="rounded-lg border border-primary/30 bg-primary/10 p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-[11px] text-primary">#{feature.feature_idx}</span>
                              {feature.attribute && (
                                <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[9px] font-medium text-primary">
                                  {feature.attribute}
                                </span>
                              )}
                            </div>
                            <span className="font-mono text-[11px] text-primary">
                              {feature.steering_strength?.toFixed(2) ?? "?"}×
                            </span>
                          </div>
                          {feature.feature_description && (
                            <p className="text-[12px] text-muted-foreground">{feature.feature_description}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border bg-card p-3">
                      <div className="mb-1 text-[10px] font-medium text-muted-foreground">GENERATED TEXT</div>
                      <p className="font-mono text-[12px] leading-relaxed">{activeResult.generate?.full_text}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <p className="text-[14px] text-muted-foreground/70">No analysis selected</p>
              <p className="mt-1 text-[12px] text-muted-foreground/50">
                Generate a response and click &quot;View analysis&quot;
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
