import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  try {
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    const backendResponse = await fetch(`${BACKEND_URL}/api/status`);
    if (!backendResponse.ok) {
      return NextResponse.json({ status: 'offline', device: 'unknown' }, { status: 503 });
    }
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ status: 'offline', device: 'unknown' }, { status: 502 });
  }
}
