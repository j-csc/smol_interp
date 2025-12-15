import type {
  GenerateRequest,
  GenerateResponse,
  SteerRequest,
  SteerResponse,
  ModelInfo,
  HealthResponse,
} from "./types";

const SERVER_URL = process.env.SERVER_URL || process.env.NEXT_PUBLIC_SERVER_URL || "";

if (!SERVER_URL) {
  console.warn("SERVER_URL is not set. API calls will fail.");
}

class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: any
  ) {
    super(message);
    this.name = "APIError";
  }
}

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${SERVER_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      error instanceof Error ? error.message : "Unknown error occurred"
    );
  }
}

export const api = {
  async health(): Promise<HealthResponse> {
    return fetchAPI<HealthResponse>("/health");
  },

  async getModelInfo(): Promise<ModelInfo> {
    return fetchAPI<ModelInfo>("/info");
  },

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    return fetchAPI<GenerateResponse>("/generate", {
      method: "POST",
      body: JSON.stringify({
        prompt: request.prompt,
        max_new_tokens: request.max_new_tokens ?? 20,
        top_k: request.top_k ?? 5,
      }),
    });
  },

  async steer(request: SteerRequest): Promise<SteerResponse> {
    return fetchAPI<SteerResponse>("/steer", {
      method: "POST",
      body: JSON.stringify({
        prompt: request.prompt,
        steering_configs: request.steering_configs,
        max_new_tokens: request.max_new_tokens ?? 100,
      }),
    });
  },
};

export { APIError };
