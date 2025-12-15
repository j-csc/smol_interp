"use client";

import * as React from "react";
import { api, APIError } from "@/lib/api";
import type { GenerateResponse, TokenExplanation } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HugeiconsIcon } from "@hugeicons/react";
import { Loading03Icon, Rocket01Icon } from "@hugeicons/core-free-icons";

export function GenerationInterface() {
  const [prompt, setPrompt] = React.useState("");
  const [maxNewTokens, setMaxNewTokens] = React.useState(20);
  const [topK, setTopK] = React.useState(5);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<GenerateResponse | null>(null);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.generate({
        prompt,
        max_new_tokens: maxNewTokens,
        top_k: topK,
      });
      setResult(response);
    } catch (err) {
      if (err instanceof APIError) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Generate with Feature Explanations</CardTitle>
          <CardDescription>
            Generate text and see which SAE features activate for each token
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleGenerate}>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="prompt">Prompt</FieldLabel>
                <Textarea
                  id="prompt"
                  placeholder="Enter your prompt here..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  required
                  rows={3}
                />
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field>
                  <FieldLabel htmlFor="max-tokens">Max New Tokens</FieldLabel>
                  <Input
                    id="max-tokens"
                    type="number"
                    min={1}
                    max={512}
                    value={maxNewTokens}
                    onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  />
                </Field>
                <Field>
                  <FieldLabel htmlFor="top-k">Top K Features</FieldLabel>
                  <Input
                    id="top-k"
                    type="number"
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                  />
                </Field>
              </div>
              <Field orientation="horizontal">
                <Button type="submit" disabled={loading || !prompt.trim()}>
                  {loading ? (
                    <>
                      <HugeiconsIcon
                        icon={Loading03Icon}
                        strokeWidth={2}
                        className="animate-spin"
                        data-icon="inline-start"
                      />
                      Generating...
                    </>
                  ) : (
                    <>
                      <HugeiconsIcon
                        icon={Rocket01Icon}
                        strokeWidth={2}
                        data-icon="inline-start"
                      />
                      Generate
                    </>
                  )}
                </Button>
              </Field>
            </FieldGroup>
          </form>
        </CardContent>
        {error && (
          <CardFooter>
            <div className="w-full rounded-md border border-red-500 bg-red-50 p-4 text-sm text-red-900 dark:bg-red-950 dark:text-red-100">
              <strong>Error:</strong> {error}
            </div>
          </CardFooter>
        )}
      </Card>

      {result && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Generated Text</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded-md bg-muted p-4 font-mono text-sm">
                {result.full_text}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Input Token Features</CardTitle>
              <CardDescription>
                Features that activated for each token in your prompt
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TokenExplanationsList
                explanations={result.input_token_explanations}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Output Token Features</CardTitle>
              <CardDescription>
                Features that predicted each generated token
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TokenExplanationsList
                explanations={result.output_token_explanations}
              />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function TokenExplanationsList({
  explanations,
}: {
  explanations: TokenExplanation[];
}) {
  return (
    <div className="space-y-4">
      {explanations.map((tokenExpl, idx) => (
        <div key={idx} className="rounded-md border p-4">
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="outline" className="font-mono">
              {tokenExpl.token}
            </Badge>
            <span className="text-sm text-muted-foreground">
              {tokenExpl.features.length} features
            </span>
          </div>
          <div className="space-y-2">
            {tokenExpl.features.map((feat, featIdx) => (
              <div
                key={featIdx}
                className="flex flex-col gap-1 rounded-sm bg-muted/50 p-2 text-sm"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">
                    Feature #{feat.feature_idx}
                  </span>
                  <Badge variant="secondary" className="text-xs">
                    {feat.activation.toFixed(2)}
                  </Badge>
                </div>
                <p className="text-muted-foreground">{feat.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
