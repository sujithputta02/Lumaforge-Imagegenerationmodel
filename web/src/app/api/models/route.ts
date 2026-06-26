import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  try {
    const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';
    const backendResponse = await fetch(`${BACKEND_URL}/api/models/available`);
    
    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: 'Failed to fetch available models' },
        { status: backendResponse.status }
      );
    }
    
    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching models:', error);
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 502 }
    );
  }
}
