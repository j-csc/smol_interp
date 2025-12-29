"use client";

import { useState } from "react";

interface TokenExplanation {
  token: string;
  features: Array<{
    feature_idx: number;
    activation: number;
    explanation: string;
  }>;
}

interface MixerPanelProps {
  data: {
    prompt: string;
    full_text: string;
    input_token_explanations: TokenExplanation[];
    output_token_explanations: TokenExplanation[];
    feature_max_activations: Record<string, number>;
  };
}

export function MixerPanel({ data }: MixerPanelProps) {
  const allFeatures = [
    ...data.input_token_explanations.flatMap((t) => t.features),
    ...data.output_token_explanations.flatMap((t) => t.features),
  ];

  const featureMap = new Map<
    number,
    { activation: number; explanation: string }
  >();

  allFeatures.forEach((f) => {
    const existing = featureMap.get(f.feature_idx);
    if (!existing || f.activation > existing.activation) {
      featureMap.set(f.feature_idx, {
        activation: f.activation,
        explanation: f.explanation,
      });
    }
  });

  const features = Array.from(featureMap.entries())
    .map(([idx, data]) => ({
      feature_idx: idx,
      activation: data.activation,
      explanation: data.explanation,
    }))
    .sort((a, b) => b.activation - a.activation)
    .slice(0, 16);

  const maxActivation = Math.max(...features.map((f) => f.activation));
  const [soloedChannel, setSoloedChannel] = useState<number | null>(null);
  const [masterGain, setMasterGain] = useState(1);

  return (
    <div className="bg-background">
      <div className="border-border bg-background rounded-sm border">
        <div className="border-border bg-muted/50 flex items-center justify-between border-b px-2 py-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-medium uppercase tracking-wider">
              Mixer
            </span>
            <span className="text-muted-foreground text-[9px]">
              {features.length} channels
            </span>
          </div>
          <button
            onClick={() => setSoloedChannel(null)}
            className="text-muted-foreground hover:text-foreground text-[9px] uppercase tracking-wider"
          >
            Clear Solo
          </button>
        </div>

        <div className="grid grid-cols-8 gap-[1px] bg-border p-[1px]">
          {features.map((feature) => (
            <MixerChannel
              key={feature.feature_idx}
              featureIdx={feature.feature_idx}
              activation={feature.activation}
              explanation={feature.explanation}
              maxActivation={maxActivation}
              isSoloed={soloedChannel === feature.feature_idx}
              isMutedBySolo={
                soloedChannel !== null && soloedChannel !== feature.feature_idx
              }
              onSolo={() =>
                setSoloedChannel(
                  soloedChannel === feature.feature_idx
                    ? null
                    : feature.feature_idx,
                )
              }
              masterGain={masterGain}
            />
          ))}
        </div>

        <div className="border-border border-t bg-muted/30 p-2">
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[10px] font-medium uppercase tracking-wider">
              Master
            </span>
            <span className="text-muted-foreground font-mono text-[9px]">
              {(masterGain * 100).toFixed(0)}%
            </span>
          </div>
          <div className="relative h-2 w-full rounded-sm bg-muted">
            <div
              className="bg-primary h-full rounded-sm transition-all"
              style={{ width: `${Math.min(masterGain * 100, 100)}%` }}
            />
            {masterGain > 1 && (
              <div className="bg-red-600 absolute right-0 top-0 h-full w-0.5 animate-pulse" />
            )}
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={masterGain}
              onChange={(e) => setMasterGain(parseFloat(e.target.value))}
              className="absolute inset-0 w-full cursor-pointer opacity-0"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

interface MixerChannelProps {
  featureIdx: number;
  activation: number;
  explanation: string;
  maxActivation: number;
  isSoloed: boolean;
  isMutedBySolo: boolean;
  onSolo: () => void;
  masterGain: number;
}

function MixerChannel({
  featureIdx,
  activation,
  explanation,
  maxActivation,
  isSoloed,
  isMutedBySolo,
  onSolo,
  masterGain,
}: MixerChannelProps) {
  const [value, setValue] = useState(activation);
  const [isMuted, setIsMuted] = useState(false);

  const effectiveValue = isMuted || isMutedBySolo ? 0 : value * masterGain;
  const percentage = (effectiveValue / maxActivation) * 100;

  const getActivationColor = () => {
    if (effectiveValue === 0) return "bg-muted";
    if (percentage > 80) return "bg-red-500";
    if (percentage > 60) return "bg-orange-500";
    if (percentage > 40) return "bg-yellow-500";
    return "bg-primary";
  };

  return (
    <div
      className={`bg-muted/30 flex flex-col p-1.5 transition-opacity ${
        isMutedBySolo ? "opacity-30" : ""
      }`}
    >
      <div className="mb-1 flex items-center justify-between gap-0.5">
        <span
          className="text-foreground truncate text-[9px] font-medium"
          title={`#${featureIdx}`}
        >
          {featureIdx}
        </span>
        <div className="flex gap-0.5">
          <button
            onClick={onSolo}
            className={`h-2 w-2 rounded-sm border transition-colors ${
              isSoloed
                ? "border-yellow-500 bg-yellow-500"
                : "border-border bg-muted hover:bg-muted-foreground"
            }`}
            title="Solo"
          />
          <button
            onClick={() => setIsMuted(!isMuted)}
            className={`h-2 w-2 rounded-sm border transition-colors ${
              isMuted ? "border-border bg-muted" : "border-primary bg-primary"
            }`}
            title="Mute"
          />
        </div>
      </div>

      <div className="relative mb-1.5 h-32 w-full" title={explanation}>
        <div className="bg-muted absolute inset-0 rounded-sm">
          <div
            className={`${getActivationColor()} absolute bottom-0 left-0 right-0 transition-all`}
            style={{ height: `${Math.min(percentage, 100)}%` }}
          />
          {percentage > 100 && (
            <div className="bg-red-600 absolute left-0 right-0 top-0 h-0.5 animate-pulse" />
          )}
        </div>
        <input
          type="range"
          min="0"
          max={maxActivation}
          step="0.1"
          value={value}
          onChange={(e) => setValue(parseFloat(e.target.value))}
          className="absolute inset-0 h-full w-full cursor-ns-resize opacity-0"
        />
      </div>

      <div className="text-muted-foreground text-center font-mono text-[9px]">
        {effectiveValue.toFixed(0)}
      </div>

      <div
        className="text-muted-foreground mt-0.5 truncate text-[8px]"
        title={explanation}
      >
        {explanation.slice(0, 15)}...
      </div>
    </div>
  );
}
