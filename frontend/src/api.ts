import type { ApiCallResult, ApiEnvelope } from './types';

type ApiMethod = 'GET' | 'POST' | 'PATCH' | 'DELETE';

interface CallOptions {
  method?: ApiMethod;
  body?: FormData | unknown;
  auth?: boolean;
}

const API_BASE = '/api/v1';

function buildHeaders(body: FormData | unknown | undefined, apiKey: string | null, auth: boolean): Headers {
  const headers = new Headers();

  if (auth) {
    const key = apiKey?.trim();
    if (key) {
      headers.set('Authorization', `Bearer ${key}`);
    }
  }

  if (body !== undefined && !(body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  return headers;
}

async function parseResponseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  return response.text();
}

export async function apiEnvelope<T>(
  apiKey: string | null,
  path: string,
  options: CallOptions = {},
): Promise<ApiCallResult<T>> {
  const method = options.method ?? 'GET';
  const auth = options.auth ?? true;

  if (auth && !(apiKey?.trim())) {
    return {
      ok: false,
      status: 0,
      error: 'API key required for this action',
    };
  }

  try {
    const body =
      options.body === undefined
        ? undefined
        : options.body instanceof FormData
          ? options.body
          : JSON.stringify(options.body);

    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: buildHeaders(options.body, apiKey, auth),
      body,
    });

    const parsed = await parseResponseBody(response);

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {
        ok: response.ok,
        status: response.status,
        error: response.ok ? undefined : String(parsed),
        raw: parsed,
      };
    }

    const envelope = parsed as ApiEnvelope<T>;

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        envelope,
        error: envelope.error?.message ?? `Request failed (${response.status})`,
      };
    }

    if (envelope.error) {
      return {
        ok: false,
        status: response.status,
        envelope,
        error: envelope.error.message,
      };
    }

    return {
      ok: true,
      status: response.status,
      envelope,
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      error: error instanceof Error ? error.message : 'Unknown network error',
    };
  }
}

export async function apiRaw<T>(
  apiKey: string | null,
  path: string,
  options: CallOptions = {},
): Promise<ApiCallResult<T>> {
  const method = options.method ?? 'GET';
  const auth = options.auth ?? true;

  if (auth && !(apiKey?.trim())) {
    return {
      ok: false,
      status: 0,
      error: 'API key required for this action',
    };
  }

  try {
    const body =
      options.body === undefined
        ? undefined
        : options.body instanceof FormData
          ? options.body
          : JSON.stringify(options.body);

    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers: buildHeaders(options.body, apiKey, auth),
      body,
    });

    const parsed = (await parseResponseBody(response)) as T;

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        raw: parsed,
        error: `Request failed (${response.status})`,
      };
    }

    return {
      ok: true,
      status: response.status,
      raw: parsed,
    };
  } catch (error) {
    return {
      ok: false,
      status: 0,
      error: error instanceof Error ? error.message : 'Unknown network error',
    };
  }
}
