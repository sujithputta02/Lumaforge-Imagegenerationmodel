import { NextRequest, NextResponse } from 'next/server';

// Zero-dependency Token Bucket Rate Limiter
const tracking = new Map<string, { tokens: number; lastRefill: number }>();
const LIMIT = 20; // 20 requests
const WINDOW = 60 * 1000; // per 60 seconds

function checkRateLimit(ip: string): { allowed: boolean; retryAfter: number } {
  const now = Date.now();
  const data = tracking.get(ip) || { tokens: LIMIT, lastRefill: now };
  
  const elapsed = now - data.lastRefill;
  const refilled = Math.floor(elapsed / (WINDOW / LIMIT));
  
  const tokens = Math.min(LIMIT, data.tokens + refilled);
  const lastRefill = refilled > 0 ? now : data.lastRefill;
  
  if (tokens <= 0) {
    const nextRefill = data.lastRefill + (WINDOW / LIMIT);
    const retryAfter = Math.max(1, Math.ceil((nextRefill - now) / 1000));
    return { allowed: false, retryAfter };
  }
  
  tracking.set(ip, { tokens: tokens - 1, lastRefill });
  return { allowed: true, retryAfter: 0 };
}

export async function POST(req: NextRequest) {
  const ip = req.headers.get('x-forwarded-for') || '127.0.0.1';
  
  // Rate limit check
  const limitCheck = checkRateLimit(ip);
  if (!limitCheck.allowed) {
    return NextResponse.json(
      {
        error: 'Too Many Requests',
        message: `Frontend rate limit exceeded. Please retry in ${limitCheck.retryAfter}s.`
      },
      { 
        status: 429,
        headers: { 'Retry-After': limitCheck.retryAfter.toString() }
      }
    );
  }
  
  try {
    const body = await req.json();
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    
    // Proxy request to Python FastAPI Backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/remove-background`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    });
    
    if (backendResponse.status === 429) {
      const data = await backendResponse.json();
      return NextResponse.json(data, { status: 429 });
    }
    
    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      return NextResponse.json(
        { error: 'Backend Error', message: errorText || 'Failed to remove background.' },
        { status: backendResponse.status }
      );
    }
    
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[API Proxy Remove Background Error]:', error);
    return NextResponse.json(
      { error: 'Connection Error', message: 'Failed to connect to LumaForge background removal backend.' },
      { status: 502 }
    );
  }
}
