import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    
    const backendResponse = await fetch(`${BACKEND_URL}/api/inpaint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    
    if (!backendResponse.ok) {
      const error = await backendResponse.json();
      return NextResponse.json(error, { status: backendResponse.status });
    }
    
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error during inpainting:', error);
    return NextResponse.json(
      { error: 'Failed to inpaint' },
      { status: 502 }
    );
  }
}
