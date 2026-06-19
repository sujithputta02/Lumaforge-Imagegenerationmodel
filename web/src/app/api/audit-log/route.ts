import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get('limit') || '20';
  
  try {
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    const backendResponse = await fetch(`${BACKEND_URL}/api/audit-log?limit=${limit}`);
    if (!backendResponse.ok) {
      return NextResponse.json({ logs: [] }, { status: 500 });
    }
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ logs: [] }, { status: 502 });
  }
}
