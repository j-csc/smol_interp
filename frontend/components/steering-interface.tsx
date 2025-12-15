"use client";

import * as React from "react";
import { api, APIError } from "@/lib/api";
import type { SteeringConfig, SteerResponse } from "@/lib/types";
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
import {
  Loading03Icon,
  Navigation04Icon,
  PlusSignIcon,
  Delete02Icon,
} from "@hugeicons/core-free-icons";

export function SteeringInterface() {
  const [prompt, setPrompt] = React.useState("");
  const [maxNewTokens, setMaxNewTokens] = React.useState(100);
  const [steeringConfigs, setSteeringConfigs] = React.useState<
    SteeringConfig[]
  >([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<SteerResponse | null>(null);

  const addSteeringConfig = () => {
    setSteeringConfigs([
      ...steeringConfigs,
      { steering_feature: 0, max_act: 30.0, steering_strength: 1.5 },
    ]);
  };

  const removeSteeringConfig = (index: number) => {
    setSteeringConfigs(steeringConfigs.filter((_, i) => i !== index));
  };

  const updateSteeringConfig = (
    index: number,
    field: keyof SteeringConfig,
    value: number
  ) => {
    const updated = [...steeringConfigs];
    updated[index] = { ...updated[index], [field]: value };
    setSteeringConfigs(updated);
  };

  const handleSteer = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.steer({
        prompt,
        steering_configs: steeringConfigs,
        max_new_tokens: maxNewTokens,
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
          <CardTitle>Generate with Steering</CardTitle>
          <CardDescription>
            Steer the model by adding scaled feature directions to its
            activations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSteer}>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="steer-prompt">Prompt</FieldLabel>
                <Textarea
                  id="steer-prompt"
                  placeholder="Enter your prompt here..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  required
                  rows={3}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="steer-max-tokens">
                  Max New Tokens
                </FieldLabel>
                <Input
                  id="steer-max-tokens"
                  type="number"
                  min={1}
                  max={512}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                />
              </Field>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <FieldLabel>Steering Configurations</FieldLabel>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={addSteeringConfig}
                  >
                    <HugeiconsIcon
                      icon={PlusSignIcon}
                      strokeWidth={2}
                      data-icon="inline-start"
                    />
                    Add Steering Vector
                  </Button>
                </div>

                {steeringConfigs.length === 0 ? (
                  <div className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
                    No steering vectors configured. Click "Add Steering Vector"
                    to get started.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {steeringConfigs.map((config, idx) => (
                      <div
                        key={idx}
                        className="rounded-md border bg-muted/50 p-4"
                      >
                        <div className="mb-3 flex items-center justify-between">
                          <Badge variant="outline">Vector {idx + 1}</Badge>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => removeSteeringConfig(idx)}
                          >
                            <HugeiconsIcon
                              icon={Delete02Icon}
                              strokeWidth={2}
                            />
                            <span className="sr-only">Remove</span>
                          </Button>
                        </div>
                        <div className="grid grid-cols-3 gap-3">
                          <Field>
                            <FieldLabel
                              htmlFor={`feature-${idx}`}
                              className="text-xs"
                            >
                              Feature Index
                            </FieldLabel>
                            <Input
                              id={`feature-${idx}`}
                              type="number"
                              min={0}
                              value={config.steering_feature}
                              onChange={(e) =>
                                updateSteeringConfig(
                                  idx,
                                  "steering_feature",
                                  Number(e.target.value)
                                )
                              }
                            />
                          </Field>
                          <Field>
                            <FieldLabel
                              htmlFor={`max-act-${idx}`}
                              className="text-xs"
                            >
                              Max Activation
                            </FieldLabel>
                            <Input
                              id={`max-act-${idx}`}
                              type="number"
                              step="0.1"
                              value={config.max_act}
                              onChange={(e) =>
                                updateSteeringConfig(
                                  idx,
                                  "max_act",
                                  Number(e.target.value)
                                )
                              }
                            />
                          </Field>
                          <Field>
                            <FieldLabel
                              htmlFor={`strength-${idx}`}
                              className="text-xs"
                            >
                              Strength
                            </FieldLabel>
                            <Input
                              id={`strength-${idx}`}
                              type="number"
                              step="0.1"
                              value={config.steering_strength}
                              onChange={(e) =>
                                updateSteeringConfig(
                                  idx,
                                  "steering_strength",
                                  Number(e.target.value)
                                )
                              }
                            />
                          </Field>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <Field orientation="horizontal">
                <Button
                  type="submit"
                  disabled={
                    loading || !prompt.trim() || steeringConfigs.length === 0
                  }
                >
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
                        icon={Navigation04Icon}
                        strokeWidth={2}
                        data-icon="inline-start"
                      />
                      Generate with Steering
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
        <Card>
          <CardHeader>
            <CardTitle>Steered Generation Result</CardTitle>
            <CardDescription>
              Text generated with{" "}
              {steeringConfigs.length === 1
                ? "1 steering vector"
                : `${steeringConfigs.length} steering vectors`}{" "}
              applied
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="mb-2 text-sm font-medium">Original Prompt:</div>
                <div className="rounded-md bg-muted/50 p-3 font-mono text-sm">
                  {result.prompt}
                </div>
              </div>
              <div>
                <div className="mb-2 text-sm font-medium">Generated Text:</div>
                <div className="rounded-md bg-muted p-4 font-mono text-sm">
                  {result.full_text}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
