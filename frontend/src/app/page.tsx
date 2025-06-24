"use client";

import React, { useState, useEffect, useRef } from 'react';

// --- Type Definitions ---
/**
 * Defines the structure for a single message in the chat history.
 */
interface Message {
  role: 'user' | 'bot';
  text: string;
}

// --- UI Icon Components ---
const SendIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const UploadIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <polyline points="17 8 12 3 7 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <line x1="12" y1="3" x2="12" y2="15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
);

const LoadingSpinner = () => (
  <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
);

// --- Main Chat Component ---
/**
 * The main chat component that handles the entire user interface,
 * state management, and API interactions for the RAG application.
 */
export default function ChatPage() {
  // State for the conversation history
  const [messages, setMessages] = useState<Message[]>([]);

  // State for the user's current input
  const [query, setQuery] = useState<string>('');

  // State for the selected PDF file
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // State for the current conversation's session ID
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);

  // State for loading and status indicators
  const [isQueryLoading, setIsQueryLoading] = useState<boolean>(false);
  const [isUploadLoading, setIsUploadLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  // Refs for direct DOM element access
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Effect to auto-scroll to the latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handler for file input changes
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
      setUploadStatus(`Selected: ${file.name}`);
      setError(null);
    } else {
      setSelectedFile(null);
      setError("Please select a valid PDF file.");
    }
  };

  // Handler for submitting the selected file to the backend
  const handleFileUpload = async () => {
    if (!selectedFile) {
        setError("Please select a file first.");
        return;
    }

    setIsUploadLoading(true);
    setUploadStatus(`Uploading "${selectedFile.name}"...`);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/upload/`;
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "File upload failed.");
        }

        const data = await response.json();
        setUploadStatus(`Success: "${data.filename}" is indexed. You can now ask questions.`);
        
        // Reset chat state after a new document is successfully indexed
        setMessages([]);
        setChatSessionId(null);

    } catch (err: any) {
        setError(err.message);
        setUploadStatus('Upload failed. Please try again.');
    } finally {
        setIsUploadLoading(false);
        setSelectedFile(null);
    }
  };

  // Handler for submitting the user's query to the backend
  const handleSubmitQuery = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Optimistically update the UI with the user's message
    setMessages(prev => [...prev, { role: 'user', text: query }]);
    setIsQueryLoading(true);
    setError(null);
    setQuery('');

    try {
      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/query/`;
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          chat_session_id: chatSessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'An unknown error occurred.');
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'bot', text: data.answer }]);
      setChatSessionId(data.chat_session_id);

    } catch (err: any) {
      setError(err.message);
      setMessages(prev => [...prev, { role: 'bot', text: "Sorry, I couldn't get a response. Please check the backend or try again." }]);
    } finally {
      setIsQueryLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white font-sans">
      
      <header className="bg-gray-800/80 backdrop-blur-sm p-4 border-b border-gray-700 shadow-lg z-10">
        <div className="max-w-5xl mx-auto flex justify-between items-center">
            <div className="flex-1">
                <h1 className="text-xl font-bold text-gray-100">Semantic Search RAG</h1>
                <p className="text-xs text-gray-400">Built with FastAPI, Next.js, and Gemini</p>
            </div>
            <div className="flex items-center space-x-3 flex-1 justify-end">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="application/pdf"
                    className="hidden"
                    disabled={isUploadLoading}
                />
                <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploadLoading}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors"
                >
                    <UploadIcon />
                    <span>Choose PDF</span>
                </button>
                {selectedFile && (
                    <button
                        onClick={handleFileUpload}
                        disabled={isUploadLoading}
                        className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-wait transition-colors font-semibold"
                    >
                       {isUploadLoading ? 'Uploading...' : 'Upload'}
                    </button>
                )}
            </div>
        </div>
        {(uploadStatus || error) && (
            <div className="max-w-5xl mx-auto mt-3 text-center text-sm">
                {error && <p className="text-red-400">{error}</p>}
                {uploadStatus && !error && <p className="text-gray-300">{uploadStatus}</p>}
            </div>
        )}
      </header>

      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
             <div className="text-center text-gray-500 mt-10">
                <p>No messages yet.</p>
                <p>Upload a PDF to get started!</p>
             </div>
          )}
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex mb-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`rounded-2xl p-3 max-w-lg lg:max-w-2xl shadow-lg animate-fade-in ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-gray-700 text-gray-200 rounded-bl-none'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.text}</p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="bg-gray-800/50 backdrop-blur-sm p-4 border-t border-gray-700">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmitQuery} className="flex items-center space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about the uploaded document..."
              className="flex-1 p-3 bg-gray-700 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-100 border border-transparent"
              disabled={isQueryLoading}
            />
            <button
              type="submit"
              className="bg-blue-600 p-3 rounded-full hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed transition-colors flex items-center justify-center w-12 h-12 shrink-0"
              disabled={isQueryLoading || !query.trim()}
            >
              {isQueryLoading ? <LoadingSpinner /> : <SendIcon />}
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}
