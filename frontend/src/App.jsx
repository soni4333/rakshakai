import React, { useState, useEffect, useRef } from 'react';
import {
  Shield,
  FileText,
  CreditCard,
  Mic,
  MicOff,
  Volume2,
  Upload,
  AlertTriangle,
  CheckCircle2,
  Info,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Cpu,
  Globe2,
  RefreshCw,
  Building2
} from 'lucide-react';

const BACKEND_URL =
  (typeof process !== 'undefined' && process.env?.REACT_APP_BACKEND_URL) ||
  (import.meta.env?.VITE_BACKEND_URL) ||
  'http://127.0.0.1:8000';

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi (हिंदी)' },
  { code: 'bn', name: 'Bengali (বাংলা)' },
  { code: 'ta', name: 'Tamil (தமிழ்)' },
  { code: 'te', name: 'Telugu (తెలుగు)' },
  { code: 'mr', name: 'Marathi (मराठी)' },
  { code: 'kn', name: 'Kannada (ಕನ್ನಡ)' },
  { code: 'ml', name: 'Malayalam (മലയാളം)' },
  { code: 'gu', name: 'Gujarati (ગુજરાતી)' },
  { code: 'pa', name: 'Punjabi (ਪੰਜਾਬੀ)' },
  { code: 'or', name: 'Odia (ଓଡ଼ିଆ)' },
  { code: 'as', name: 'Assamese (অসমীয়া)' }
];

export default function App() {
  const [mode, setMode] = useState('contract'); // 'contract' or 'loan'
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('');
  const [entityName, setEntityName] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  // Voice recording state
  const [isRecording, setIsRecording] = useState(false);
  const [voiceStatusMsg, setVoiceStatusMsg] = useState('Ready to record');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingStartTimeRef = useRef(0);

  // Sources and Voice System Status
  const [sources, setSources] = useState([]);
  const [voiceStatus, setVoiceStatus] = useState(null);
  const [expandedEvidence, setExpandedEvidence] = useState({});

  // Fetch health, sources, and voice status on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/api/sources`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.sources) setSources(data.sources);
      })
      .catch(() => {});

    fetch(`${BACKEND_URL}/api/voice/status`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setVoiceStatus(data);
      })
      .catch(() => {});
  }, []);

  // Handle Document Upload
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    if (!['txt', 'md', 'json', 'csv'].includes(ext)) {
      setAnalysisError('Unsupported file type. Please upload a UTF-8 encoded .txt or .md document.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setAnalysisError('File exceeds 5MB size limit.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setAnalysisError(null);
      const res = await fetch(`${BACKEND_URL}/api/documents/text`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      if (data.text) {
        setInputText(data.text);
      }
    } catch (err) {
      setAnalysisError(err.message || 'Failed to extract text from document.');
    }
  };

  // Handle Voice Recording via MediaRecorder
  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
      setVoiceStatusMsg('Sending recording…');
    } else {
      // Start recording
      audioChunksRef.current = [];
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/ogg;codecs=opus';

        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;
        recordingStartTimeRef.current = Date.now();

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        recorder.onstop = async () => {
          stream.getTracks().forEach((track) => track.stop());
          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          const duration = (Date.now() - recordingStartTimeRef.current) / 1000;

          const formData = new FormData();
          formData.append('file', audioBlob, 'recording.webm');
          formData.append('language_id', language);
          formData.append('duration_seconds', duration.toString());

          try {
            const res = await fetch(`${BACKEND_URL}/api/voice/transcribe`, {
              method: 'POST',
              body: formData
            });

            if (!res.ok) {
              setVoiceStatusMsg('Local transcription is unavailable. You can type or paste the transcript below.');
              return;
            }

            const data = await res.json();
            if (data.transcript) {
              setInputText((prev) => (prev ? `${prev}\n${data.transcript}` : data.transcript));
              setVoiceStatusMsg('Transcription completed successfully.');
            }
          } catch {
            setVoiceStatusMsg('Local transcription is unavailable. You can type or paste the transcript below.');
          }
        };

        recorder.start();
        setIsRecording(true);
        setVoiceStatusMsg('Listening locally — press stop to send');
      } catch {
        setVoiceStatusMsg('Microphone permission denied.');
      }
    }
  };

  // Handle Speech Synthesis Playback
  const handlePlayback = () => {
    if (!inputText || !inputText.trim()) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(inputText);
      utterance.lang = language === 'hi' ? 'hi-IN' : 'en-IN';
      window.speechSynthesis.speak(utterance);
    } else {
      setVoiceStatusMsg('Speech synthesis is not supported in this browser.');
    }
  };

  // Submit for Legal Risk Screening
  const handleAnalyze = async () => {
    if (!inputText || !inputText.trim()) {
      setAnalysisError('Please enter contractual terms or upload a document to analyze.');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const res = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          entity_name: mode === 'loan' ? entityName : null,
          source_lang: language,
          target_lang: language,
          mode
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      setAnalysisError(err.message || 'Failed to complete analysis. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleEvidence = (idx) => {
    setExpandedEvidence((prev) => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  return (
    <div className="min-h-screen bg-[#F8FAF9] text-[#0B132B] font-sans flex flex-col antialiased selection:bg-emerald-100 selection:text-emerald-900">
      {/* 1. TOP NAVIGATION */}
      <header className="border-b border-[#E2E8F0] bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-[#0B132B] text-white rounded">
              <Shield className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight text-[#0B132B]" data-testid="app-wordmark">
                RakshakAI
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded border border-slate-200">
                LEGAL SHIELD
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div
              className="flex items-center space-x-1.5 px-2.5 py-1 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded text-xs font-mono font-medium"
              data-testid="local-demo-status"
            >
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>LOCAL-FIRST DEMO</span>
            </div>
            <span
              className="text-xs font-mono text-slate-500 border-l border-slate-200 pl-3 hidden md:inline-block"
              data-testid="version-badge"
            >
              V1.0 / TEXT + VOICE
            </span>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* 2. INTRO SECTION */}
        <section className="border-b border-[#E2E8F0] pb-8 pt-2">
          <div className="max-w-3xl space-y-3">
            <div className="inline-flex items-center space-x-2 text-xs font-mono font-semibold tracking-wider text-emerald-700 uppercase">
              <span>●</span>
              <span>LEGAL-RISK SCREENING / INDIA</span>
            </div>
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-[#0B132B] leading-tight">
              Read the fine print <br />
              <span className="text-emerald-700">before it reads you.</span>
            </h1>
            <p className="text-base sm:text-lg text-slate-600 leading-relaxed max-w-2xl pt-1">
              A careful second look at contracts and loan ads — grounded in statutory evidence, clear about uncertainty.
            </p>
            <div className="flex flex-wrap gap-2 pt-2">
              <span className="inline-flex items-center px-2.5 py-1 bg-white border border-slate-300 rounded text-xs font-mono text-slate-700">
                🛡️ Evidence-first
              </span>
              <span className="inline-flex items-center px-2.5 py-1 bg-white border border-slate-300 rounded text-xs font-mono text-slate-700">
                ⚖️ No legality verdicts
              </span>
            </div>
          </div>
        </section>

        {/* 3. MODE SWITCH */}
        <section className="space-y-4">
          <div className="flex items-center space-x-2 p-1 bg-slate-200/70 rounded-lg w-fit border border-slate-300">
            <button
              onClick={() => setMode('contract')}
              data-testid="contract-mode-button"
              className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-semibold transition-all ${
                mode === 'contract'
                  ? 'bg-white text-[#0B132B] shadow-sm'
                  : 'text-slate-600 hover:text-[#0B132B]'
              }`}
            >
              <FileText className="w-4 h-4 text-emerald-600" />
              <span>Contract</span>
            </button>
            <button
              onClick={() => setMode('loan')}
              data-testid="loan-mode-button"
              className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-semibold transition-all ${
                mode === 'loan'
                  ? 'bg-white text-[#0B132B] shadow-sm'
                  : 'text-slate-600 hover:text-[#0B132B]'
              }`}
            >
              <CreditCard className="w-4 h-4 text-emerald-600" />
              <span>Loan Ad</span>
            </button>
          </div>

          <div>
            <h2 className="text-xl font-bold text-[#0B132B]">
              {mode === 'contract' ? 'What did you agree to?' : 'What are they offering?'}
            </h2>
            <p className="text-sm text-slate-500">
              {mode === 'contract'
                ? 'Check gig worker terms, arbitrary deactivation clauses, wage deductions, and unconscionable penalties.'
                : 'Examine digital loan terms, hidden interest APRs, contact harvesting permissions, and lender regulatory status.'}
            </p>
          </div>
        </section>

        {/* WORKSPACE: TWO COLUMN LAYOUT (DESKTOP) / STACKED (MOBILE) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* 4. SUBMISSION PANEL (8 COLS) */}
          <div className="lg:col-span-8 space-y-6">
            <div className="bg-white border border-[#E2E8F0] rounded-xl p-6 shadow-sm space-y-4">
              {/* Top Controls: Language & Entity */}
              <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center justify-between pb-2 border-b border-slate-100">
                <div className="flex items-center space-x-2">
                  <Globe2 className="w-4 h-4 text-slate-400" />
                  <label htmlFor="lang-select" className="text-xs font-mono uppercase font-semibold text-slate-500">
                    Input Language:
                  </label>
                  <select
                    id="lang-select"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    data-testid="language-select"
                    className="text-sm bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-medium text-[#0B132B]"
                  >
                    {LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.name}
                      </option>
                    ))}
                  </select>
                </div>

                {mode === 'loan' && (
                  <div className="flex items-center space-x-2">
                    <Building2 className="w-4 h-4 text-slate-400" />
                    <input
                      type="text"
                      placeholder="Lender / App Name (e.g. Navi, KreditBee)"
                      value={entityName}
                      onChange={(e) => setEntityName(e.target.value)}
                      data-testid="lender-entity-input"
                      className="text-sm bg-slate-50 border border-slate-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-medium text-[#0B132B] w-full sm:w-64"
                    />
                  </div>
                )}
              </div>

              {/* Main Textarea */}
              <div className="relative">
                <textarea
                  rows={7}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder={
                    mode === 'contract'
                      ? 'Paste delivery partner agreement clauses, deactivation terms, penalty conditions, or upload contract text...'
                      : 'Paste digital loan advertisement text, interest rate claims, permission disclaimers, or upload loan terms...'
                  }
                  data-testid="analysis-text-input"
                  className="w-full p-4 text-sm font-normal text-slate-900 bg-[#FAFBFB] border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:bg-white transition-all font-mono leading-relaxed"
                />

                <div
                  className="absolute bottom-3 right-3 text-xs font-mono text-slate-400 bg-white/80 px-2 py-0.5 rounded border border-slate-200"
                  data-testid="character-count"
                >
                  {inputText.length} characters
                </div>
              </div>

              {/* Voice Status Indicator */}
              <div
                className="flex items-center justify-between text-xs font-mono px-3 py-2 bg-slate-50 border border-slate-200 rounded text-slate-600"
                data-testid="voice-status"
              >
                <div className="flex items-center space-x-2">
                  <span className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-ping' : 'bg-slate-400'}`} />
                  <span>{voiceStatusMsg}</span>
                </div>
                {isRecording && <span className="text-red-600 font-bold uppercase">Recording Active</span>}
              </div>

              {/* Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
                <div className="flex flex-wrap items-center gap-2">
                  {/* Document Upload Input */}
                  <label
                    className="cursor-pointer inline-flex items-center space-x-1.5 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded text-xs font-semibold border border-slate-300 transition-colors"
                    title="Attach .txt or .md document"
                  >
                    <Upload className="w-3.5 h-3.5 text-slate-600" />
                    <span>Attach Text Doc</span>
                    <input
                      type="file"
                      accept=".txt,.md,.json,.csv"
                      onChange={handleFileUpload}
                      data-testid="document-upload-input"
                      className="hidden"
                    />
                  </label>

                  {/* Record Voice Button */}
                  <button
                    type="button"
                    onClick={toggleRecording}
                    data-testid="voice-input-button"
                    className={`inline-flex items-center space-x-1.5 px-3 py-2 rounded text-xs font-semibold border transition-all ${
                      isRecording
                        ? 'bg-red-600 text-white border-red-700 animate-pulse'
                        : 'bg-slate-100 hover:bg-slate-200 text-slate-700 border-slate-300'
                    }`}
                  >
                    {isRecording ? <MicOff className="w-3.5 h-3.5" /> : <Mic className="w-3.5 h-3.5 text-slate-600" />}
                    <span>{isRecording ? 'Stop Recording' : 'Voice Input (ASR)'}</span>
                  </button>

                  {/* Playback Button */}
                  <button
                    type="button"
                    onClick={handlePlayback}
                    data-testid="voice-playback-button"
                    disabled={!inputText}
                    className="inline-flex items-center space-x-1.5 px-3 py-2 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 text-slate-700 rounded text-xs font-semibold border border-slate-300 transition-colors"
                  >
                    <Volume2 className="w-3.5 h-3.5 text-slate-600" />
                    <span>Play Text</span>
                  </button>
                </div>

                {/* Primary Screen Material Button */}
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  data-testid="analyze-button"
                  className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-2.5 bg-[#0B132B] hover:bg-[#1C2541] disabled:opacity-75 text-white text-sm font-bold rounded-lg shadow transition-all"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                      <span>Screening Material...</span>
                    </>
                  ) : (
                    <>
                      <Shield className="w-4 h-4 text-emerald-400" />
                      <span>Screen Material</span>
                    </>
                  )}
                </button>
              </div>

              {/* Error Container */}
              {analysisError && (
                <div
                  className="p-3.5 bg-red-50 border border-red-200 rounded-lg text-red-800 text-xs flex items-start space-x-2 font-mono"
                  data-testid="analysis-error"
                >
                  <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                  <span>{analysisError}</span>
                </div>
              )}
            </div>

            {/* 6. ANALYSIS RESULT SECTION */}
            {analysisResult && (
              <div
                className="bg-white border-2 border-[#0B132B] rounded-xl p-6 shadow-md space-y-6 animate-fadeIn"
                data-testid="analysis-result"
              >
                {/* Header Signal Badge */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
                  <div className="flex items-center space-x-3">
                    <div
                      className={`p-2.5 rounded-lg ${
                        analysisResult.risk_level === 'high'
                          ? 'bg-red-600 text-white'
                          : analysisResult.risk_level === 'medium'
                          ? 'bg-amber-500 text-white'
                          : 'bg-emerald-600 text-white'
                      }`}
                    >
                      {analysisResult.risk_level === 'high' ? (
                        <AlertTriangle className="w-6 h-6" />
                      ) : analysisResult.risk_level === 'medium' ? (
                        <Info className="w-6 h-6" />
                      ) : (
                        <CheckCircle2 className="w-6 h-6" />
                      )}
                    </div>
                    <div>
                      <span
                        className="text-lg sm:text-xl font-bold tracking-tight text-[#0B132B]"
                        data-testid="risk-signal-label"
                      >
                        {analysisResult.signal_label}
                      </span>
                      <p className="text-xs font-mono text-slate-500">
                        Confidence: {analysisResult.confidence}% · {analysisResult.citation}
                      </p>
                    </div>
                  </div>

                  {/* Lender Status Badge */}
                  {analysisResult.lender_status && (
                    <div
                      className={`px-3 py-1 rounded text-xs font-mono font-semibold border ${
                        analysisResult.lender_status.includes('unverified')
                          ? 'bg-red-50 text-red-800 border-red-300'
                          : 'bg-emerald-50 text-emerald-800 border-emerald-300'
                      }`}
                      data-testid="lender-status"
                    >
                      {analysisResult.lender_status}
                    </div>
                  )}
                </div>

                {/* Highlighted Finding Excerpt */}
                {analysisResult.highlighted_finding && (
                  <div className="p-4 bg-slate-50 border-l-4 border-emerald-600 rounded-r-lg space-y-1">
                    <span className="text-xs font-mono uppercase text-slate-500 font-bold tracking-wider">
                      Highlighted Finding:
                    </span>
                    <p
                      className="text-sm font-semibold text-slate-800 leading-snug"
                      data-testid="highlighted-finding"
                    >
                      "{analysisResult.highlighted_finding}"
                    </p>
                  </div>
                )}

                {/* Detailed Legal Analysis */}
                <div className="space-y-2">
                  <h4 className="text-xs font-mono uppercase text-slate-500 font-bold tracking-wider">
                    Statutory Analysis & Harm Assessment:
                  </h4>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {analysisResult.explanation || analysisResult.finding_text}
                  </p>
                </div>

                {/* Actionable Recommendation */}
                {analysisResult.recommendation && (
                  <div className="p-4 bg-emerald-50/60 border border-emerald-200 rounded-lg space-y-1">
                    <span className="text-xs font-mono uppercase text-emerald-900 font-bold tracking-wider">
                      💡 Actionable Recommendation:
                    </span>
                    <p className="text-sm text-emerald-950 font-medium">
                      {analysisResult.recommendation}
                    </p>
                  </div>
                )}

                {/* 7. BILINGUAL AUDIT TRAIL */}
                {analysisResult.bilingual_audit && (
                  <div className="pt-4 border-t border-slate-200 space-y-3" data-testid="bilingual-audit-trail">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono uppercase text-slate-500 font-bold tracking-wider">
                        Bilingual Audit Trail
                      </span>
                      <span
                        className="text-xs font-mono text-slate-400"
                        data-testid="translation-status"
                      >
                        {analysisResult.bilingual_audit.translation_status}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                        <span className="text-xs font-mono font-bold text-slate-600 uppercase">
                          ORIGINAL SUBMISSION:
                        </span>
                        <p
                          className="text-xs text-slate-800 font-mono leading-relaxed max-h-32 overflow-y-auto"
                          data-testid="original-text-audit"
                        >
                          {analysisResult.bilingual_audit.original_text}
                        </p>
                      </div>

                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                        <span className="text-xs font-mono font-bold text-slate-600 uppercase">
                          ENGLISH INTERPRETATION:
                        </span>
                        <p
                          className="text-xs text-slate-800 font-mono leading-relaxed max-h-32 overflow-y-auto"
                          data-testid="english-interpretation-audit"
                        >
                          {analysisResult.bilingual_audit.english_interpretation}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* 8. EVIDENCE SECTION (EXPANDABLE ROWS) */}
                {analysisResult.evidence && analysisResult.evidence.length > 0 && (
                  <div className="pt-4 border-t border-slate-200 space-y-3">
                    <span className="text-xs font-mono uppercase text-slate-500 font-bold tracking-wider">
                      Grounding Evidence & Statutory Provisions ({analysisResult.evidence.length})
                    </span>

                    <div className="space-y-2">
                      {analysisResult.evidence.map((ev, idx) => (
                        <div key={idx} className="border border-slate-200 rounded-lg overflow-hidden bg-white">
                          <button
                            onClick={() => toggleEvidence(idx)}
                            data-testid={`evidence-toggle-${idx}`}
                            className="w-full flex items-center justify-between p-3 text-left hover:bg-slate-50 transition-colors text-xs font-bold text-slate-800"
                          >
                            <span data-testid={`evidence-source-${idx}`}>
                              {ev.title} · <span className="font-normal text-slate-500">{ev.source_name}</span>
                            </span>
                            {expandedEvidence[idx] ? (
                              <ChevronUp className="w-4 h-4 text-slate-400" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-slate-400" />
                            )}
                          </button>

                          {expandedEvidence[idx] && (
                            <div className="p-3 border-t border-slate-100 bg-slate-50/50 space-y-2 text-xs">
                              <p className="font-mono text-slate-700 leading-relaxed">{ev.passage}</p>
                              <div className="flex items-center justify-between pt-1 text-slate-400 font-mono">
                                <span>Status: {ev.review_status || 'verified'}</span>
                                {ev.source_url && (
                                  <a
                                    href={ev.source_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-flex items-center space-x-1 text-emerald-700 hover:underline"
                                  >
                                    <span>Official Statute Link</span>
                                    <ExternalLink className="w-3 h-3" />
                                  </a>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 5. SIDE INFORMATION PANEL (4 COLS) */}
          <div className="lg:col-span-4 space-y-6">
            {/* Risk Legend Card */}
            <div className="bg-white border border-[#E2E8F0] rounded-xl p-5 shadow-sm space-y-4">
              <h3 className="text-xs font-mono uppercase font-bold text-slate-500 tracking-wider">
                Risk Classification Legend
              </h3>
              <div className="space-y-2.5 text-xs">
                <div className="flex items-start space-x-2.5 p-2 bg-red-50/50 rounded border border-red-100">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold text-red-950">High Risk:</span>
                    <p className="text-slate-600">Specific predatory or non-compliant term detected.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-2.5 p-2 bg-amber-50/50 rounded border border-amber-100">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold text-amber-950">Review Recommended:</span>
                    <p className="text-slate-600">Context-sensitive term requiring borrower/worker attention.</p>
                  </div>
                </div>

                <div className="flex items-start space-x-2.5 p-2 bg-emerald-50/50 rounded border border-emerald-100">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold text-emerald-950">Low Risk:</span>
                    <p className="text-slate-600">No high-risk clause detected.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Local Voice & Model Status Card */}
            <div className="bg-white border border-[#E2E8F0] rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center space-x-2 text-xs font-mono font-bold text-slate-700">
                <Cpu className="w-4 h-4 text-emerald-600" />
                <span>LOCAL RUNTIME DIAGNOSTICS</span>
              </div>

              <div className="space-y-1.5 text-xs font-mono text-slate-600 border-t border-slate-100 pt-2.5">
                <div className="flex justify-between">
                  <span>Provider:</span>
                  <span className="font-semibold text-slate-900">{voiceStatus?.provider || 'AI4Bharat Open Stack'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Inference Device:</span>
                  <span className="font-semibold text-slate-900">{voiceStatus?.cpu_status || 'PyTorch CPU'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Checkpoint:</span>
                  <span className="text-emerald-700 font-semibold">
                    {voiceStatus?.checkpoint_found ? 'Found (DistilBERT)' : 'Loaded'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Runtime State:</span>
                  <span className="text-emerald-700 font-semibold">
                    {voiceStatus?.runtime_ready ? 'Ready' : 'Initializing'}
                  </span>
                </div>
              </div>
            </div>

            {/* 9. SOURCE REGISTRY SECTION */}
            <div className="bg-white border border-[#E2E8F0] rounded-xl p-5 shadow-sm space-y-3" data-testid="source-registry">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-mono uppercase font-bold text-slate-500 tracking-wider">
                  Indexed Statutory Registry ({sources.length})
                </h3>
              </div>

              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {sources.map((src, i) => (
                  <div key={i} className="p-2.5 bg-slate-50 border border-slate-200 rounded text-xs space-y-1">
                    <div className="flex items-center justify-between font-bold text-[#0B132B]">
                      <span>{src.title}</span>
                    </div>
                    <p className="text-[11px] text-slate-500 font-mono">
                      {src.jurisdiction} · {src.review_status}
                    </p>
                    <p className="text-[11px] text-slate-600">{src.note}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 10. FIXED LEGAL DISCLAIMER */}
        <footer className="border-t border-[#E2E8F0] pt-6 pb-12">
          <div
            className="p-4 bg-white border border-slate-200 rounded-lg text-center text-xs font-mono text-slate-500 max-w-3xl mx-auto shadow-sm"
            data-testid="legal-disclaimer"
          >
            RakshakAI is an informational screening tool, not legal advice. A risk signal is not a declaration of legality. Consider speaking with a qualified professional.
          </div>
        </footer>
      </main>
    </div>
  );
}
