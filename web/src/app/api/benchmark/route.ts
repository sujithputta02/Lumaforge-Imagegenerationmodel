import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    const backendResponse = await fetch(`${BACKEND_URL}/api/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: 'Connection Error', message: 'Failed to run benchmarks.' }, { status: 502 });
  }
}
