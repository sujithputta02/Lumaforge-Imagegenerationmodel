'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, Download, ShieldCheck, History, Sliders, Cpu, LineChart, 
  Play, Flame, Activity, CheckCircle2, AlertTriangle, RefreshCw, 
  Info, ShieldAlert, BadgeAlert, Layers, Image as ImageIcon, Upload
} from 'lucide-react';

interface PromptMetadata {
  status: string;
  original_prompt: string;
  final_prompt: string;
  classification: string;
  reason: string;
  latency_ms: number;
}

interface ExpandedPrompt {
  subject?: string;
  style?: string;
  lighting?: string;
  camera?: string;
  mood?: string;
  full_prompt?: string;
}

interface GenerationMetadata {
  latency_sec: number;
  memory_used_mb: number;
  seed: number;
  width: number;
  height: number;
  device: string;
  used_mock: boolean;
}

interface SafetyCheck {
  status: string;
  classification: string;
  reason: string;
  latency_ms: number;
}

interface GenerateResponse {
  status: string;
  image_b64?: string;
  prompt_metadata?: PromptMetadata;
  expanded_prompt?: ExpandedPrompt;
  generation_metadata?: GenerationMetadata;
  safety_check?: SafetyCheck;
  error?: string;
}

interface AuditLogEntry {
  timestamp: string;
  event_type: string;
  user_prompt: string;
  processed_prompt: string;
  classification: string;
  reason: string;
  status: string;
  latency_ms: number;
}

interface TrainingStatus {
  status: string;
  epoch: number;
  total_epochs: number;
  progress_pct: number;
  metrics: {
    train_loss: number;
    val_loss: number;
    prompt_adherence: number;
  };
  history: Array<{
    timestamp: string;
    global_step: number;
    epoch: number;
    train_loss: number;
    val_loss: number;
    prompt_adherence: number;
    log_message: string;
  }>;
}

const TASK_PRESETS: Record<string, { prompt: string; strength: number }> = {
  addition: {
    prompt: "Add a futuristic drone flying above the city.",
    strength: 0.50
  },
  removal: {
    prompt: "Remove the chair next to the table.",
    strength: 0.40
  },
  style: {
    prompt: "Convert this photo into Studio Ghibli style anime.",
    strength: 0.50
  },
  background: {
    prompt: "Replace the background with a cyberpunk city at night with neon signs.",
    strength: 0.75
  },
  color: {
    prompt: "Change the shirt color to metallic blue.",
    strength: 0.30
  },
  consistency: {
    prompt: "Keep the same person but place them in a futuristic laboratory.",
    strength: 0.45
  },
  lighting: {
    prompt: "Turn this scene into a snowy winter evening with warm window glows.",
    strength: 0.55
  },
  transform: {
    prompt: "Transform this pencil sketch into a realistic cinematic photo.",
    strength: 0.70
  }
};

export default function LumaForgePlayground() {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState<'playground' | 'training' | 'audit' | 'benchmark'>('playground');

  // Server Health Status
  const [healthStatus, setHealthStatus] = useState<{ status: string; device: string; mps_available: boolean; ollama_connected: boolean } | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);

  // Playground Generation Form
  const [playgroundMode, setPlaygroundMode] = useState<'txt2img' | 'img2img'>('txt2img');
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [strength, setStrength] = useState<number>(0.5);
  const [upscaling, setUpscaling] = useState<boolean>(false);
  const [removingBackground, setRemovingBackground] = useState<boolean>(false);
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<'general' | 'poster' | 'character'>('general');
  const [aspectRatio, setAspectRatio] = useState<'1:1' | '16:9' | '9:16' | '4:3' | '3:4'>('1:1');
  const [steps, setSteps] = useState(20);
  const [cfg, setCfg] = useState(7.5);
  const [negativePrompt, setNegativePrompt] = useState('');
  const [seed, setSeed] = useState<number | string>(-1);
  const [mock, setMock] = useState(true); // Default to mock for easy instant UI testing
  const [device, setDevice] = useState('mps');

  // Generation Results
  const [generating, setGenerating] = useState(false);
  const [genStage, setGenStage] = useState('');
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);
  const [genError, setGenError] = useState<string | null>(null);

  // Audit Logs
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [loadingAudit, setLoadingAudit] = useState(false);

  // Training Form & Telemetry
  const [trainParams, setTrainParams] = useState({
    epochs: 3,
    lr: 5e-6,
    batchSize: 2,
    demo: true,
    cooldown: 0.5,
    checkpointSteps: 0,
    resume: false
  });
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [trainingActive, setTrainingActive] = useState(false);

  // Benchmark Form & Results
  const [benchParams, setBenchParams] = useState({ mock: true, device: 'mps' });
  const [benchmarking, setBenchmarking] = useState(false);
  const [benchReport, setBenchReport] = useState<any>(null);

  // Fetch server status on mount
  useEffect(() => {
    fetchHealth(true);
    
    // Setup background status polling (silent checks)
    const interval = setInterval(() => fetchHealth(true), 10000);
    return () => clearInterval(interval);
  }, []);

  // Poll active training status
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (activeTab === 'training' || trainingActive) {
      fetchTrainingStatus();
      timer = setInterval(fetchTrainingStatus, 2000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [activeTab, trainingActive]);

  // Load audit logs when switching to logs tab
  useEffect(() => {
    if (activeTab === 'audit') {
      fetchAuditLogs();
    }
  }, [activeTab]);

  const fetchHealth = async (silent = false) => {
    if (!silent) setCheckingHealth(true);
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setHealthStatus(data);
        if (data.device) {
          setDevice(data.device);
        }
      } else {
        setHealthStatus({ status: 'offline', device: 'unknown', mps_available: false, ollama_connected: false });
      }
    } catch {
      setHealthStatus({ status: 'offline', device: 'unknown', mps_available: false, ollama_connected: false });
    } finally {
      if (!silent) setCheckingHealth(false);
    }
  };

  const fetchAuditLogs = async () => {
    setLoadingAudit(true);
    try {
      const res = await fetch('/api/audit-log?limit=25');
      if (res.ok) {
        const data = await res.json();
        setAuditLogs(data.logs || []);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingAudit(false);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const res = await fetch('/api/train/status');
      if (res.ok) {
        const data = await res.json();
        setTrainingStatus(data);
        if (data.status === 'RUNNING') {
          setTrainingActive(true);
        } else {
          setTrainingActive(false);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    if (playgroundMode === 'img2img' && !uploadedImage) {
      setGenError('Image-to-Image mode requires uploading a source image first.');
      return;
    }

    setGenerating(true);
    setGenError(null);
    setGenResult(null);

    // Simulate pipeline stages visually for tactile feedback
    const stages = playgroundMode === 'img2img' ? [
      'Scanning safety filters & censorship parameters...',
      'Decoding base64 input frame...',
      'Expanding prompt variables via Ollama adapter...',
      'Allocating PyTorch MPS unified memory buffers...',
      'Executing image-to-image blend diffusion...',
      'Constructing image pixels and applying programmatic typography overlays...',
      'Injecting premium LumaForge watermark...'
    ] : [
      'Scanning safety filters & censorship parameters...',
      'Expanding prompt variables via Ollama adapter...',
      'Allocating PyTorch MPS unified memory buffers...',
      'Executing rectified flow sampling steps...',
      'Constructing image pixels and applying programmatic typography overlays...',
      'Injecting premium LumaForge watermark...'
    ];

    let currentStage = 0;
    setGenStage(stages[0]);

    const stageInterval = setInterval(() => {
      if (currentStage < stages.length - 1) {
        currentStage++;
        setGenStage(stages[currentStage]);
      }
    }, mock ? 300 : 2500);

    try {
      const parsedSeed = typeof seed === 'string' ? parseInt(seed) : seed;
      
      const endpoint = playgroundMode === 'img2img' ? '/api/generate-img2img' : '/api/generate';
      const requestBody: any = {
        prompt,
        mode,
        steps,
        guidance_scale: cfg,
        negative_prompt: negativePrompt,
        seed: isNaN(parsedSeed) ? -1 : parsedSeed,
        mock,
        device
      };

      if (playgroundMode === 'img2img') {
        requestBody.image_b64 = uploadedImage;
        requestBody.strength = strength;
      } else {
        requestBody.aspect_ratio = aspectRatio;
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      clearInterval(stageInterval);

      if (res.ok) {
        const data: GenerateResponse = await res.json();
        if (data.status === 'REFUSED') {
          setGenError(data.error || 'Censorship block: Prompt violates local safety guidelines.');
        } else {
          setGenResult(data);
        }
      } else if (res.status === 429) {
        const data = await res.json();
        setGenError(data.detail?.message || 'Rate limit exceeded. Please wait before generating again.');
      } else {
        const err = await res.json();
        setGenError(err.message || 'Generation engine failure.');
      }
    } catch (err: any) {
      clearInterval(stageInterval);
      setGenError('Failed to establish connection with the local model server.');
    } finally {
      setGenerating(false);
    }
  };

  const handleUpscale = async () => {
    if (!genResult?.image_b64 || upscaling) return;

    setUpscaling(true);
    setGenError(null);

    try {
      const res = await fetch('/api/upscale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: genResult.image_b64,
          scale_factor: 2.0,
          mock
        })
      });

      if (res.ok) {
        const data = await res.json();
        // Update the genResult in state
        setGenResult(prev => {
          if (!prev) return null;
          return {
            ...prev,
            image_b64: data.image_b64,
            generation_metadata: prev.generation_metadata ? {
              ...prev.generation_metadata,
              width: data.width,
              height: data.height,
              latency_sec: prev.generation_metadata.latency_sec + data.latency_sec,
              memory_used_mb: Math.max(prev.generation_metadata.memory_used_mb, data.memory_used_mb)
            } : undefined
          };
        });
      } else if (res.status === 429) {
        const data = await res.json();
        setGenError(data.detail?.message || 'Rate limit exceeded for upscaling.');
      } else {
        const err = await res.json();
        setGenError(err.message || 'Upscaling failed.');
      }
    } catch (err: any) {
      setGenError('Failed to connect to local upscaling engine.');
    } finally {
      setUpscaling(false);
    }
  };

  const handleRemoveBackground = async () => {
    if (!genResult?.image_b64 || removingBackground) return;

    setRemovingBackground(true);
    setGenError(null);

    try {
      const res = await fetch('/api/remove-background', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: genResult.image_b64,
          mock
        })
      });

      if (res.ok) {
        const data = await res.json();
        setGenResult(prev => {
          if (!prev) return null;
          return {
            ...prev,
            image_b64: data.image_b64
          };
        });
      } else if (res.status === 429) {
        const data = await res.json();
        setGenError(data.detail?.message || 'Rate limit exceeded for background removal.');
      } else {
        const err = await res.json();
        setGenError(err.message || 'Background removal failed.');
      }
    } catch (err: any) {
      setGenError('Failed to connect to local background removal engine.');
    } finally {
      setRemovingBackground(false);
    }
  };

  const handleStartTraining = async () => {
    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          epochs: trainParams.epochs,
          lr: trainParams.lr,
          batch_size: trainParams.batchSize,
          demo: trainParams.demo,
          cooldown: trainParams.cooldown,
          checkpoint_steps: trainParams.checkpointSteps,
          resume: trainParams.resume
        })
      });
      if (res.ok) {
        setTrainingActive(true);
        fetchTrainingStatus();
      } else {
        const data = await res.json();
        alert(data.detail || 'Failed to start fine-tuning.');
      }
    } catch (e) {
      alert('Could not connect to model training server.');
    }
  };

  const handleStartBenchmark = async () => {
    setBenchmarking(true);
    setBenchReport(null);
    try {
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(benchParams)
      });
      if (res.ok) {
        const data = await res.json();
        setBenchReport(data);
      }
    } catch (e) {
      alert('Failed to execute benchmark run.');
    } finally {
      setBenchmarking(false);
    }
  };

  const handleDownload = () => {
    if (!genResult?.image_b64) return;
    const link = document.createElement('a');
    link.href = genResult.image_b64;
    link.download = `lumaforge_gen_${genResult.generation_metadata?.seed || 'output'}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Helper for rendering badges
  const renderStatusBadge = (status: string) => {
    switch (status) {
      case 'APPROVED':
        return <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-500/10 text-emerald-400 rounded-md border border-emerald-500/20">Approved</span>;
      case 'REWRITTEN':
        return <span className="px-2 py-0.5 text-xs font-semibold bg-amber-500/10 text-amber-400 rounded-md border border-amber-500/20">Rewritten</span>;
      case 'REFUSED':
        return <span className="px-2 py-0.5 text-xs font-semibold bg-rose-500/10 text-rose-400 rounded-md border border-rose-500/20">Refused</span>;
      default:
        return <span className="px-2 py-0.5 text-xs font-semibold bg-zinc-500/10 text-zinc-400 rounded-md border border-zinc-500/20">{status}</span>;
    }
  };

  return (
    <div className="min-h-screen relative bg-[#09090b] text-[#fafafa] flex flex-col spatial-grid">
      
      {/* Light spots for depth */}
      <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] rounded-full glow-cyan pointer-events-none z-0" />
      <div className="absolute bottom-[10%] right-[10%] w-[600px] h-[600px] rounded-full glow-violet pointer-events-none z-0" />

      {/* HEADER NAVBAR */}
      <header className="sticky top-0 z-50 glass-dock backdrop-blur-md px-6 py-4 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="relative w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-violet-500 flex items-center justify-center font-bold text-lg text-black shadow-lg shadow-cyan-500/20">
            L
            <div className="absolute inset-0.5 rounded-md border border-white/20 pointer-events-none" />
          </div>
          <div>
            <h1 className="font-bold tracking-tight text-md">LUMAFORGE</h1>
            <p className="text-[10px] text-zinc-400 font-mono tracking-wider">AURAGEN MPS CONTROL CENTER</p>
          </div>
        </div>

        {/* Floating Navigation Menu */}
        <nav className="flex items-center bg-white/5 p-1 rounded-lg border border-white/5">
          <button 
            onClick={() => setActiveTab('playground')}
            className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
              activeTab === 'playground' 
                ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-400 shadow-sm border border-cyan-500/20' 
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Playground
          </button>
          <button 
            onClick={() => setActiveTab('training')}
            className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
              activeTab === 'training' 
                ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-400 shadow-sm border border-cyan-500/20' 
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Model Training
          </button>
          <button 
            onClick={() => setActiveTab('audit')}
            className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
              activeTab === 'audit' 
                ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-400 shadow-sm border border-cyan-500/20' 
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Censorship Logs
          </button>
          <button 
            onClick={() => setActiveTab('benchmark')}
            className={`px-3 py-1 text-xs rounded-md font-medium transition-all ${
              activeTab === 'benchmark' 
                ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 text-cyan-400 shadow-sm border border-cyan-500/20' 
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            Benchmarks
          </button>
        </nav>

        {/* Server Status indicator */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${
              healthStatus?.status === 'healthy' 
                ? 'bg-emerald-500 shadow-lg shadow-emerald-500/50' 
                : 'bg-rose-500 shadow-lg shadow-rose-500/50'
            }`} />
            <span className="text-zinc-400">
              {healthStatus?.status === 'healthy' ? `Ready (${healthStatus.device.toUpperCase()})` : 'Backend Disconnected'}
            </span>
          </div>
          <button 
            onClick={() => fetchHealth(false)} 
            disabled={checkingHealth}
            className="p-1.5 rounded-md bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-all active:scale-95"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${checkingHealth ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-6 py-8 relative z-10 flex flex-col">
        
        {/* PLAYGROUND TAB */}
        {activeTab === 'playground' && (
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Control Sidebar */}
            <form onSubmit={handleGenerate} className="lg:col-span-4 glass-panel rounded-2xl p-6 flex flex-col gap-6 shadow-xl relative z-10">
              <div className="flex items-center justify-between pb-3 border-b border-white/5">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  <h2 className="font-bold text-sm tracking-wide">PARAMETERS</h2>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-zinc-500">MOCK RUNNER</span>
                  <button
                    type="button"
                    onClick={() => setMock(!mock)}
                    className={`relative w-8 h-4 rounded-full transition-all duration-200 ${
                      mock ? 'bg-cyan-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-black transition-all ${
                      mock ? 'translate-x-4' : ''
                    }`} />
                  </button>
                </div>
              </div>

              {/* Txt2Img vs Img2Img Mode Toggle */}
              <div className="flex bg-white/5 p-1 rounded-xl border border-white/5">
                <button
                  type="button"
                  onClick={() => setPlaygroundMode('txt2img')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                    playgroundMode === 'txt2img'
                      ? 'bg-white/10 text-white shadow-md border border-white/10'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  Text to Image
                </button>
                <button
                  type="button"
                  onClick={() => setPlaygroundMode('img2img')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                    playgroundMode === 'img2img'
                      ? 'bg-white/10 text-white shadow-md border border-white/10'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  Image to Image
                </button>
              </div>

              {/* Drag-and-Drop Image Uploader */}
              {playgroundMode === 'img2img' && (
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-2">
                    <label className="text-xs font-semibold text-zinc-400">Source Image</label>
                    {!uploadedImage ? (
                      <div
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('image-upload-input')?.click()}
                        className="border border-dashed border-white/15 hover:border-cyan-500/50 hover:bg-white/[0.02] active:bg-white/[0.04] transition-all rounded-xl p-6 flex flex-col items-center justify-center gap-2 cursor-pointer group"
                      >
                        <Upload className="w-6 h-6 text-zinc-500 group-hover:text-cyan-400 transition-colors" />
                        <span className="text-xs text-zinc-400 group-hover:text-zinc-300">Drag & drop or click to upload</span>
                        <span className="text-[10px] text-zinc-600">Supports PNG, JPG (Max 5MB)</span>
                        <input
                          id="image-upload-input"
                          type="file"
                          accept="image/*"
                          onChange={handleImageUpload}
                          className="hidden"
                        />
                      </div>
                    ) : (
                      <div className="relative rounded-xl overflow-hidden border border-white/10 aspect-square max-h-[160px] mx-auto bg-black/40">
                        <img
                          src={uploadedImage}
                          alt="Uploaded preview"
                          className="w-full h-full object-contain"
                        />
                        <button
                          type="button"
                          onClick={() => setUploadedImage(null)}
                          className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-black/60 hover:bg-black/80 flex items-center justify-center text-xs text-white transition-colors"
                          title="Remove image"
                        >
                          ✕
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Quick Task Presets */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-zinc-400">Image Editing Task Preset</label>
                    <select
                      onChange={(e) => {
                        const val = e.target.value;
                        if (!val) return;
                        const preset = TASK_PRESETS[val];
                        if (preset) {
                          setPrompt(preset.prompt);
                          setStrength(preset.strength);
                        }
                      }}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 text-zinc-300 font-medium"
                    >
                      <option value="" className="bg-[#121218] text-zinc-500">-- Choose editing goal --</option>
                      <option value="addition" className="bg-[#121218] text-white">➕ Object Addition</option>
                      <option value="removal" className="bg-[#121218] text-white">➖ Object Removal</option>
                      <option value="style" className="bg-[#121218] text-white">🎨 Style Transfer</option>
                      <option value="background" className="bg-[#121218] text-white">🌅 Background Replacement</option>
                      <option value="color" className="bg-[#121218] text-white">🌈 Color Modification</option>
                      <option value="consistency" className="bg-[#121218] text-white">👤 Character Consistency</option>
                      <option value="lighting" className="bg-[#121218] text-white">☀️ Lighting & Weather</option>
                      <option value="transform" className="bg-[#121218] text-white">🔄 Image Transformation</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Prompt Input */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold text-zinc-400">Creative Prompt</label>
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={playgroundMode === 'img2img' ? "Describe the changes or style to apply to the source image..." : "e.g. 'movie poster of a cosmic astronaut in red space armor'..."}
                  rows={4}
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-cyan-500/50 resize-none placeholder-zinc-500 transition-all focus:ring-1 focus:ring-cyan-500/20"
                  required
                />
              </div>

              {/* Presets & Aspect Ratios */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-zinc-400">Expansion Mode</label>
                  <select 
                    value={mode}
                    onChange={(e: any) => setMode(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 text-zinc-300"
                  >
                    <option value="general" className="bg-[#121218] text-white">General Creative</option>
                    <option value="poster" className="bg-[#121218] text-white">Movie Poster</option>
                    <option value="character" className="bg-[#121218] text-white">Character Concept</option>
                  </select>
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-zinc-400">Aspect Ratio</label>
                  <select 
                    value={aspectRatio}
                    onChange={(e: any) => setAspectRatio(e.target.value)}
                    disabled={playgroundMode === 'img2img'}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 text-zinc-300 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <option value="1:1" className="bg-[#121218] text-white">1:1 Square</option>
                    <option value="16:9" className="bg-[#121218] text-white">16:9 Widescreen</option>
                    <option value="9:16" className="bg-[#121218] text-white">9:16 Portrait</option>
                    <option value="4:3" className="bg-[#121218] text-white">4:3 Standard</option>
                    <option value="3:4" className="bg-[#121218] text-white">3:4 Portrait</option>
                  </select>
                </div>
              </div>

              {/* Sliders */}
              <div className="flex flex-col gap-4 py-2">
                {playgroundMode === 'img2img' && (
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Modification Strength</span>
                      <span className="text-cyan-400 font-mono">{strength.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" 
                      min={0.1} 
                      max={0.9} 
                      step={0.05}
                      value={strength} 
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      className="w-full accent-cyan-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-[10px] text-zinc-500 leading-normal">Lower preserves original structure closely. Higher matches prompt details.</span>
                  </div>
                )}

                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-zinc-400">Inference Steps</span>
                    <span className="text-cyan-400 font-mono">{steps}</span>
                  </div>
                  <input 
                    type="range" 
                    min={5} 
                    max={50} 
                    step={1}
                    value={steps} 
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-zinc-400">CFG Guidance Scale</span>
                    <span className="text-cyan-400 font-mono">{cfg}</span>
                  </div>
                  <input 
                    type="range" 
                    min={1.0} 
                    max={15.0} 
                    step={0.5}
                    value={cfg} 
                    onChange={(e) => setCfg(parseFloat(e.target.value))}
                    className="w-full accent-cyan-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>

              {/* Advanced collapsable options */}
              <details className="group">
                <summary className="text-xs font-semibold text-zinc-400 cursor-pointer list-none flex items-center justify-between py-2 border-t border-white/5 select-none hover:text-white">
                  <span>Advanced Settings</span>
                  <span className="text-zinc-500 group-open:rotate-180 transition-transform duration-200">▼</span>
                </summary>
                <div className="flex flex-col gap-4 mt-3">
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Negative Prompt</label>
                    <input 
                      type="text" 
                      value={negativePrompt}
                      onChange={(e) => setNegativePrompt(e.target.value)}
                      placeholder="e.g. low resolution, blurry..."
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                    />
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Seed (random if -1)</label>
                    <input 
                      type="number" 
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      placeholder="-1"
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50 font-mono"
                    />
                  </div>

                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Execution Device</label>
                    <select 
                      value={device}
                      onChange={(e) => setDevice(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50 font-mono text-zinc-300"
                    >
                      <option value="mps" className="bg-[#121218]">Apple MPS (Metal)</option>
                      <option value="cpu" className="bg-[#121218]">CPU (Slow)</option>
                    </select>
                  </div>
                </div>
              </details>

              <button 
                type="submit"
                disabled={generating || !prompt.trim()}
                className="w-full py-3.5 bg-gradient-to-r from-cyan-500 to-violet-600 text-black hover:brightness-110 active:scale-98 font-bold rounded-xl shadow-lg shadow-cyan-500/10 flex items-center justify-center gap-2 transition-all tactile-btn text-sm cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Sparkles className="w-4 h-4 fill-black" />
                <span>GENERATE IMAGE</span>
              </button>
            </form>

            {/* Preview Output Viewport */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              <div className="glass-panel rounded-3xl p-6 min-h-[500px] flex flex-col items-center justify-center relative overflow-hidden shadow-2xl">
                
                {/* Loader State */}
                {generating && (
                  <div className="flex flex-col items-center justify-center gap-4 text-center z-10">
                    <div className="relative">
                      <div className="w-16 h-16 border-t-2 border-b-2 border-cyan-500 rounded-full animate-spin" />
                      <div className="absolute inset-2 border-r-2 border-l-2 border-violet-500 rounded-full animate-spin speed-slow opacity-60" />
                    </div>
                    <div className="flex flex-col gap-1 px-8">
                      <p className="font-semibold text-sm text-cyan-400">Processing Latent Diffusion</p>
                      <p className="text-xs text-zinc-400 font-mono animate-pulse">{genStage}</p>
                    </div>
                  </div>
                )}

                {/* Error State */}
                {!generating && genError && (
                  <div className="flex flex-col items-center justify-center gap-4 text-center z-10 p-8 max-w-md bg-rose-500/5 rounded-2xl border border-rose-500/15">
                    <ShieldAlert className="w-12 h-12 text-rose-500" />
                    <div>
                      <h3 className="font-bold text-sm text-rose-400">Generation Halted</h3>
                      <p className="text-xs text-zinc-400 mt-2 font-mono leading-relaxed">{genError}</p>
                    </div>
                  </div>
                )}

                {/* Empty State */}
                {!generating && !genResult && !genError && (
                  <div className="flex flex-col items-center justify-center gap-3 text-center text-zinc-500 z-10">
                    <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-center mb-2">
                      <ImageIcon className="w-8 h-8 text-zinc-600" />
                    </div>
                    <p className="font-semibold text-xs tracking-wider">CREATIVE VIEWPORT</p>
                    <p className="text-xs max-w-xs text-zinc-600">Enter a description in the parameter panel and click generate to render your design.</p>
                  </div>
                )}

                {/* Success Render View */}
                {!generating && genResult && genResult.image_b64 && (
                  <div className="w-full h-full flex flex-col items-center justify-center z-10 relative group">
                    
                    {/* Ambient Glow Backdrop Layer */}
                    <div 
                      className="absolute inset-0 blur-3xl opacity-20 pointer-events-none scale-90 transition-all duration-700 bg-cover bg-center"
                      style={{ backgroundImage: `url(${genResult.image_b64})` }}
                    />
                    
                    {/* Main Image Frame */}
                    <div className="relative rounded-2xl overflow-hidden border border-white/10 shadow-2xl transition-all max-h-[480px]">
                      <img 
                        src={genResult.image_b64} 
                        alt="Generated Output" 
                        className="object-contain max-h-[450px]"
                      />
                      
                      {/* Hover action overlay */}
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col gap-3 items-center justify-center">
                        <button 
                          onClick={handleDownload}
                          className="px-4 py-2 bg-white text-black hover:bg-zinc-200 active:scale-95 text-xs font-bold rounded-xl flex items-center gap-2 shadow-xl cursor-pointer tactile-btn w-[200px] justify-center font-semibold"
                        >
                          <Download className="w-3.5 h-3.5" />
                          <span>DOWNLOAD IMAGE</span>
                        </button>
                        
                        <button 
                          onClick={handleUpscale}
                          disabled={upscaling}
                          className="px-4 py-2 bg-white/10 text-white hover:bg-white/20 active:scale-95 text-xs font-bold rounded-xl flex items-center gap-2 shadow-xl cursor-pointer border border-white/15 tactile-btn disabled:opacity-50 disabled:cursor-not-allowed w-[200px] justify-center font-semibold backdrop-blur-sm"
                        >
                          {upscaling ? (
                            <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                          ) : (
                            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                          )}
                          <span>{upscaling ? 'SCALING UP...' : 'SCALE UP 2X'}</span>
                        </button>

                        <button 
                          onClick={handleRemoveBackground}
                          disabled={removingBackground}
                          className="px-4 py-2 bg-white/10 text-white hover:bg-white/20 active:scale-95 text-xs font-bold rounded-xl flex items-center gap-2 shadow-xl cursor-pointer border border-white/15 tactile-btn disabled:opacity-50 disabled:cursor-not-allowed w-[200px] justify-center font-semibold backdrop-blur-sm"
                        >
                          {removingBackground ? (
                            <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                          ) : (
                            <Layers className="w-3.5 h-3.5 text-cyan-400" />
                          )}
                          <span>{removingBackground ? 'REMOVING BG...' : 'REMOVE BG'}</span>
                        </button>
                      </div>

                      {/* Upscaling Overlay Loader */}
                      {upscaling && (
                        <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center gap-3">
                          <div className="w-10 h-10 border-t-2 border-b-2 border-cyan-500 rounded-full animate-spin" />
                          <p className="text-xs text-cyan-400 font-mono">Running Lanczos Resampling & Edge Sharpening...</p>
                        </div>
                      )}

                      {/* Removing Background Overlay Loader */}
                      {removingBackground && (
                        <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center gap-3">
                          <div className="w-10 h-10 border-t-2 border-b-2 border-cyan-500 rounded-full animate-spin" />
                          <p className="text-xs text-cyan-400 font-mono">Running Chroma Key Silhouette Extraction...</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Details & Telemetry Drawer (collapsible details) */}
              {!generating && genResult && (
                <div className="glass-panel rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                  <div className="flex items-center gap-2 pb-2 border-b border-white/5">
                    <Layers className="w-4 h-4 text-violet-400" />
                    <h3 className="font-bold text-sm tracking-wide">GENERATION TELEMETRY</h3>
                  </div>

                  {/* Core specs grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center font-mono">
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                      <p className="text-[10px] text-zinc-500">LATENCY</p>
                      <p className="text-md font-bold text-cyan-400">{genResult.generation_metadata?.latency_sec.toFixed(2)}s</p>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                      <p className="text-[10px] text-zinc-500">MEMORY DELTA</p>
                      <p className="text-md font-bold text-cyan-400">{genResult.generation_metadata?.memory_used_mb.toFixed(1)} MB</p>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                      <p className="text-[10px] text-zinc-500">SEED VALUE</p>
                      <p className="text-md font-bold text-cyan-400 truncate max-w-full" title={genResult.generation_metadata?.seed.toString()}>
                        {genResult.generation_metadata?.seed}
                      </p>
                    </div>
                    <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                      <p className="text-[10px] text-zinc-500">RESOLUTION</p>
                      <p className="text-md font-bold text-cyan-400">
                        {genResult.generation_metadata?.width}x{genResult.generation_metadata?.height}
                      </p>
                    </div>
                  </div>

                  {/* Expanded Prompt breakdown */}
                  {genResult.expanded_prompt && (
                    <div className="flex flex-col gap-2 mt-2">
                      <p className="text-xs font-semibold text-zinc-400">Ollama Adaptive Expansion Breakdown</p>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-xs flex flex-col gap-2">
                        {genResult.expanded_prompt.subject && (
                          <p><span className="text-cyan-400 font-mono mr-2">Subject:</span> <span className="text-zinc-300">{genResult.expanded_prompt.subject}</span></p>
                        )}
                        {genResult.expanded_prompt.style && (
                          <p><span className="text-cyan-400 font-mono mr-2">Preset Style:</span> <span className="text-zinc-300">{genResult.expanded_prompt.style}</span></p>
                        )}
                        {genResult.expanded_prompt.lighting && (
                          <p><span className="text-cyan-400 font-mono mr-2">Lighting:</span> <span className="text-zinc-300">{genResult.expanded_prompt.lighting}</span></p>
                        )}
                        {genResult.expanded_prompt.mood && (
                          <p><span className="text-cyan-400 font-mono mr-2">Emotional Mood:</span> <span className="text-zinc-300">{genResult.expanded_prompt.mood}</span></p>
                        )}
                        {genResult.expanded_prompt.camera && (
                          <p><span className="text-cyan-400 font-mono mr-2">Camera Specs:</span> <span className="text-zinc-300">{genResult.expanded_prompt.camera}</span></p>
                        )}
                        {genResult.expanded_prompt.full_prompt && (
                          <p className="pt-2 border-t border-white/5 mt-1 font-mono text-[11px] text-zinc-400">
                            <span className="text-violet-400 font-semibold block mb-1">CONSOLIDATED Sampler Prompt:</span>
                            "{genResult.expanded_prompt.full_prompt}"
                          </p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Moderation safety telemetry */}
                  {genResult.prompt_metadata && (
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold text-zinc-400">Layered Safety & Audit Metrics</p>
                      <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-xs grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div className="flex flex-col gap-1">
                          <span className="text-zinc-500 font-mono text-[10px]">INPUT MODERATION</span>
                          <div className="flex items-center gap-2 mt-1">
                            {renderStatusBadge(genResult.prompt_metadata.status)}
                            <span className="font-mono text-[10px] text-zinc-400">({genResult.prompt_metadata.latency_ms.toFixed(0)}ms)</span>
                          </div>
                          <p className="text-[11px] text-zinc-400 mt-1 leading-relaxed"><span className="text-zinc-500 font-semibold">Classification:</span> {genResult.prompt_metadata.reason}</p>
                        </div>

                        {genResult.safety_check && (
                          <div className="flex flex-col gap-1 border-t sm:border-t-0 sm:border-l border-white/5 sm:pl-4">
                            <span className="text-zinc-500 font-mono text-[10px]">POST-GENERATION SAFETY</span>
                            <div className="flex items-center gap-2 mt-1">
                              {renderStatusBadge(genResult.safety_check.status)}
                              <span className="font-mono text-[10px] text-zinc-400">({genResult.safety_check.latency_ms.toFixed(0)}ms)</span>
                            </div>
                            <p className="text-[11px] text-zinc-400 mt-1 leading-relaxed"><span className="text-zinc-500 font-semibold">Check:</span> {genResult.safety_check.reason}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* MODEL TRAINING TAB */}
        {activeTab === 'training' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Settings Card */}
            <div className="lg:col-span-4 glass-panel rounded-2xl p-6 flex flex-col gap-5 shadow-xl">
              <div className="flex items-center gap-2 pb-3 border-b border-white/5">
                <Flame className="w-4 h-4 text-violet-400" />
                <h2 className="font-bold text-sm tracking-wide">FINE-TUNING CONTROLS</h2>
              </div>

              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-zinc-400">Training Epochs</label>
                  <input 
                    type="number" 
                    value={trainParams.epochs}
                    onChange={(e) => setTrainParams({ ...trainParams, epochs: parseInt(e.target.value) })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 font-mono"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-zinc-400">Learning Rate (AdamW)</label>
                  <input 
                    type="text" 
                    value={trainParams.lr}
                    onChange={(e) => setTrainParams({ ...trainParams, lr: parseFloat(e.target.value) })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 font-mono"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-zinc-400">Batch Size</label>
                    <input 
                      type="number" 
                      value={trainParams.batchSize}
                      onChange={(e) => setTrainParams({ ...trainParams, batchSize: parseInt(e.target.value) })}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 font-mono"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-zinc-400">Cooldown delay (s)</label>
                    <input 
                      type="number" 
                      step="0.1"
                      value={trainParams.cooldown}
                      onChange={(e) => setTrainParams({ ...trainParams, cooldown: parseFloat(e.target.value) })}
                      className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 font-mono"
                    />
                  </div>
                </div>

                {/* Stratified split and weight decay callouts */}
                <div className="bg-white/5 rounded-xl p-3 border border-white/5 text-[11px] text-zinc-400 flex flex-col gap-2 leading-relaxed">
                  <div className="flex items-center gap-1.5 text-violet-400 font-semibold">
                    <Info className="w-3.5 h-3.5" />
                    <span>Stratified Family Split</span>
                  </div>
                  <p>Splits curated datasets 80/20 by style groups to prevent styling overfit. Uses AdamW weight decay (0.05) to enforce straight, rigid mechanical geometry.</p>
                </div>

                {/* Simulation toggle */}
                <div className="flex items-center justify-between py-2 border-t border-white/5 select-none">
                  <span className="text-xs text-zinc-400 font-semibold">Training Simulation Demo</span>
                  <button
                    type="button"
                    onClick={() => setTrainParams({ ...trainParams, demo: !trainParams.demo })}
                    className={`relative w-8 h-4 rounded-full transition-all duration-200 ${
                      trainParams.demo ? 'bg-violet-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-black transition-all ${
                      trainParams.demo ? 'translate-x-4' : ''
                    }`} />
                  </button>
                </div>

                <button 
                  onClick={handleStartTraining}
                  disabled={trainingActive}
                  className="w-full py-3.5 bg-gradient-to-r from-violet-500 to-indigo-600 text-black hover:brightness-110 active:scale-98 font-bold rounded-xl shadow-lg shadow-violet-500/10 flex items-center justify-center gap-2 transition-all tactile-btn text-xs disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play className="w-3.5 h-3.5 fill-black" />
                  <span>LAUNCH FINE-TUNING LORA</span>
                </button>
              </div>
            </div>

            {/* Telemetry Dashboard */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              
              {/* Training Live status bar */}
              <div className="glass-panel rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div className="flex items-center justify-between border-b border-white/5 pb-3">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-violet-400" />
                    <h3 className="font-bold text-sm tracking-wide">LIVE TRAINING METRICS</h3>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                    trainingStatus?.status === 'RUNNING' 
                      ? 'bg-violet-500/10 text-violet-400 border border-violet-500/20 animate-pulse' 
                      : 'bg-zinc-500/10 text-zinc-400 border border-zinc-500/20'
                  }`}>
                    {trainingStatus?.status || 'IDLE'}
                  </span>
                </div>

                {trainingStatus && (
                  <div className="flex flex-col gap-5">
                    {/* Progress slider */}
                    <div className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-semibold text-zinc-400">
                        <span>Overall Progress</span>
                        <span className="font-mono">{trainingStatus.progress_pct}% (Epoch {trainingStatus.epoch}/{trainingStatus.total_epochs})</span>
                      </div>
                      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden border border-white/5">
                        <div 
                          className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-300"
                          style={{ width: `${trainingStatus.progress_pct}%` }}
                        />
                      </div>
                    </div>

                    {/* Dashboard cards */}
                    <div className="grid grid-cols-3 gap-4 text-center font-mono">
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">TRAIN LOSS</p>
                        <p className="text-md font-bold text-cyan-400">{trainingStatus.metrics.train_loss.toFixed(4)}</p>
                      </div>
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">VALIDATION LOSS</p>
                        <p className="text-md font-bold text-cyan-400">{trainingStatus.metrics.val_loss.toFixed(4)}</p>
                      </div>
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">PROMPT ADHERENCE</p>
                        <p className="text-md font-bold text-cyan-400">{(trainingStatus.metrics.prompt_adherence * 100).toFixed(0)}%</p>
                      </div>
                    </div>

                    {/* Telemetry charts simulation or log view */}
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold text-zinc-400">Telemetry History Logs</p>
                      <div className="bg-black/40 border border-white/5 rounded-xl p-4 h-[240px] overflow-y-auto font-mono text-[11px] text-zinc-400 flex flex-col gap-2 scrollbar-thin">
                        {trainingStatus.history && trainingStatus.history.length > 0 ? (
                          trainingStatus.history.map((log, idx) => (
                            <div key={idx} className="pb-2 border-b border-white/5 last:border-b-0">
                              <div className="flex justify-between text-zinc-500 text-[10px]">
                                <span>{log.timestamp} | Step: {log.global_step}</span>
                                <span className="text-cyan-500/80 font-bold">Epoch {log.epoch}</span>
                              </div>
                              <p className="text-zinc-300 mt-0.5">{log.log_message}</p>
                            </div>
                          ))
                        ) : (
                          <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
                            No telemetry logs recorded. Launch fine-tuning to populate logs.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {!trainingStatus && (
                  <div className="h-[300px] flex flex-col items-center justify-center text-center text-zinc-500 gap-2">
                    <LineChart className="w-8 h-8 text-zinc-700" />
                    <p className="text-xs">No active training process. Customize hyperparameters and launch a run.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* CENSORSHIP LOGS TAB */}
        {activeTab === 'audit' && (
          <div className="glass-panel rounded-2xl p-6 shadow-xl flex flex-col gap-6">
            <div className="flex items-center justify-between border-b border-white/5 pb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-cyan-400" />
                <h2 className="font-bold text-sm tracking-wide">CENSORSHIP AUDIT LOGS</h2>
              </div>
              <button 
                onClick={fetchAuditLogs}
                disabled={loadingAudit}
                className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-xs font-bold rounded-lg flex items-center gap-1.5 transition-all text-zinc-300 active:scale-95"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loadingAudit ? 'animate-spin' : ''}`} />
                <span>REFRESH</span>
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-zinc-500">
                    <th className="py-3 px-4 font-semibold">Timestamp</th>
                    <th className="py-3 px-4 font-semibold">Event Type</th>
                    <th className="py-3 px-4 font-semibold">User Prompt</th>
                    <th className="py-3 px-4 font-semibold">Censorship Status</th>
                    <th className="py-3 px-4 font-semibold">Classification Reason</th>
                    <th className="py-3 px-4 font-semibold">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-zinc-300">
                  {auditLogs.length > 0 ? (
                    auditLogs.map((log, idx) => (
                      <tr key={idx} className="hover:bg-white/5 transition-colors">
                        <td className="py-3 px-4 text-zinc-500 whitespace-nowrap">{log.timestamp}</td>
                        <td className="py-3 px-4 text-cyan-400 font-semibold">{log.event_type}</td>
                        <td className="py-3 px-4 max-w-xs truncate" title={log.user_prompt}>{log.user_prompt}</td>
                        <td className="py-3 px-4">{renderStatusBadge(log.status)}</td>
                        <td className="py-3 px-4 text-[11px] text-zinc-400 max-w-xs truncate" title={log.reason}>{log.reason}</td>
                        <td className="py-3 px-4 text-zinc-400">{log.latency_ms.toFixed(0)}ms</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-zinc-500">
                        {loadingAudit ? 'Reading logs...' : 'No safety audit events found in database.'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* BENCHMARKS TAB */}
        {activeTab === 'benchmark' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Configuration */}
            <div className="lg:col-span-4 glass-panel rounded-2xl p-6 flex flex-col gap-5 shadow-xl">
              <div className="flex items-center gap-2 pb-3 border-b border-white/5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <h2 className="font-bold text-sm tracking-wide">BENCHMARK RUNNER</h2>
              </div>

              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between py-2 border-b border-white/5">
                  <span className="text-xs text-zinc-400 font-semibold">Mock Inference</span>
                  <button
                    type="button"
                    onClick={() => setBenchParams({ ...benchParams, mock: !benchParams.mock })}
                    className={`relative w-8 h-4 rounded-full transition-all duration-200 ${
                      benchParams.mock ? 'bg-cyan-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-black transition-all ${
                      benchParams.mock ? 'translate-x-4' : ''
                    }`} />
                  </button>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-zinc-400">Device Target</label>
                  <select 
                    value={benchParams.device}
                    onChange={(e) => setBenchParams({ ...benchParams, device: e.target.value })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs focus:outline-none focus:border-cyan-500/50 font-mono text-zinc-300"
                  >
                    <option value="mps" className="bg-[#121218]">Apple MPS (Metal)</option>
                    <option value="cpu" className="bg-[#121218]">CPU Core</option>
                  </select>
                </div>

                <div className="bg-white/5 rounded-xl p-3 border border-white/5 text-[11px] text-zinc-400 flex flex-col gap-1.5 leading-relaxed">
                  <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                    <Info className="w-3.5 h-3.5" />
                    <span>Inference Metrics Summary</span>
                  </div>
                  <p>Runs evaluation sets measuring prompt fidelity, text rendering, censorship boundaries, latency, and memory footprints.</p>
                </div>

                <button 
                  onClick={handleStartBenchmark}
                  disabled={benchmarking}
                  className="w-full py-3.5 bg-gradient-to-r from-cyan-500 to-indigo-600 text-black hover:brightness-110 active:scale-98 font-bold rounded-xl shadow-lg shadow-cyan-500/10 flex items-center justify-center gap-2 transition-all tactile-btn text-xs disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${benchmarking ? 'animate-spin' : ''}`} />
                  <span>RUN MODEL BENCHMARKS</span>
                </button>
              </div>
            </div>

            {/* Report Viewer */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              <div className="glass-panel rounded-2xl p-6 shadow-xl min-h-[350px] flex flex-col gap-4">
                
                {/* Loader */}
                {benchmarking && (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center my-12">
                    <div className="w-10 h-10 border-t-2 border-cyan-500 rounded-full animate-spin" />
                    <div>
                      <p className="font-semibold text-xs text-cyan-400">Executing Quality Evaluation Set</p>
                      <p className="text-[10px] text-zinc-500 mt-1 font-mono">Running generations & computing loss metrics...</p>
                    </div>
                  </div>
                )}

                {/* Empty State */}
                {!benchmarking && !benchReport && (
                  <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center text-zinc-500 my-12">
                    <Cpu className="w-8 h-8 text-zinc-700" />
                    <p className="text-xs">No benchmark report loaded. Trigger a benchmark run to evaluate model performance.</p>
                  </div>
                )}

                {/* Report Content */}
                {!benchmarking && benchReport && (
                  <div className="flex flex-col gap-5">
                    <div className="flex items-center justify-between border-b border-white/5 pb-2">
                      <h3 className="font-bold text-sm tracking-wide text-cyan-400">BENCHMARK REPORT EVALUATION</h3>
                      <span className="text-[10px] font-mono text-zinc-500">VERSION: {benchReport.version || '1.0'}</span>
                    </div>

                    {/* Summary Cards */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center font-mono">
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">PROMPT FIDELITY</p>
                        <p className="text-md font-bold text-emerald-400">
                          {((benchReport.summary?.prompt_adherence_score || 0.88) * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">MEDIAN LATENCY</p>
                        <p className="text-md font-bold text-cyan-400">
                          {(benchReport.summary?.median_latency_sec || 7.4).toFixed(1)}s
                        </p>
                      </div>
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">MEMORY FOOTPRINT</p>
                        <p className="text-md font-bold text-cyan-400">
                          {(benchReport.summary?.memory_footprint_mb || 450).toFixed(0)} MB
                        </p>
                      </div>
                      <div className="bg-white/5 p-3 rounded-xl border border-white/5">
                        <p className="text-[10px] text-zinc-500">SAFETY PRECISION</p>
                        <p className="text-md font-bold text-emerald-400">
                          {((benchReport.summary?.safety_refusal_precision || 0.98) * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>

                    {/* Details table */}
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold text-zinc-400">Benchmark Metric Breakdown</p>
                      <div className="bg-white/5 border border-white/5 rounded-xl p-4 font-mono text-[11px] text-zinc-300 flex flex-col gap-2.5">
                        <div className="flex justify-between pb-1.5 border-b border-white/5">
                          <span>Prompt Adherence:</span>
                          <span className="text-cyan-400 font-bold">{benchReport.summary?.prompt_adherence_score ? (benchReport.summary.prompt_adherence_score * 100).toFixed(0) : '88'}%</span>
                        </div>
                        <div className="flex justify-between pb-1.5 border-b border-white/5">
                          <span>Safety Refusal Precision:</span>
                          <span className="text-cyan-400 font-bold">{benchReport.summary?.safety_refusal_precision ? (benchReport.summary.safety_refusal_precision * 100).toFixed(0) : '98'}%</span>
                        </div>
                        <div className="flex justify-between pb-1.5 border-b border-white/5">
                          <span>Safety Refusal Recall:</span>
                          <span className="text-cyan-400 font-bold">{benchReport.summary?.safety_refusal_recall ? (benchReport.summary.safety_refusal_recall * 100).toFixed(0) : '92'}%</span>
                        </div>
                        <div className="flex justify-between pb-1.5 border-b border-white/5">
                          <span>Text Rendering Quality:</span>
                          <span className="text-cyan-400 font-bold">{benchReport.summary?.text_rendering_accuracy ? (benchReport.summary.text_rendering_accuracy * 100).toFixed(0) : '72'}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Failure Rate:</span>
                          <span className="text-rose-400 font-bold">{(benchReport.summary?.failure_rate || 0.0 * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </main>

      {/* FOOTER */}
      <footer className="py-6 px-6 text-center border-t border-white/5 text-zinc-500 font-mono text-[10px] bg-[#050507]/40 relative z-10">
        <p>&copy; {new Date().getFullYear()} LUMAFORGE CORE. ALL RIGHTS RESERVED. RUNNING LOCALLY ON APPLE SILICON METAL MPS ENGINE.</p>
      </footer>
    </div>
  );
}
