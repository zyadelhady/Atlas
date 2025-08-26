"use client";
import { useState, FormEvent } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/prism";

interface Message {
  text: string;
  isUser: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const suggestionQuestions = [
    "How do I create a pandas DataFrame?",
    "What is a pandas Series?",
    "How do I select data from a DataFrame?",
    "How do I handle missing data in pandas?",
  ];

  const handleSuggestionClick = (question: string) => {
    setInput(question);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { text: input, isUser: true };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/ai", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input, history: messages }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get reader from response body");
      }

      let receivedText = "";
      const decoder = new TextDecoder();
      setMessages((prev) => [...prev, { text: "", isUser: false }]); // Add an empty message for AI response

      let firstChunk = true;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (firstChunk) {
          setIsLoading(false); // Hide typing indicator on first chunk
          firstChunk = false;
        }
        receivedText += decoder.decode(value, { stream: true });
        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            ...newMessages[newMessages.length - 1],
            text: receivedText,
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
      const errorMessage: Message = {
        text: "Sorry, something went wrong. Please try again.",
        isUser: false,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessageContent = (text: string) => {
    const parts = text.split(/(```[\s\S]*?```)/g);
    return parts.map((part, index) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        const code = part.slice(3, -3).trim();
        const languageMatch = code.match(/^(\w+)\n/);
        const language = languageMatch ? languageMatch[1] : "python"; // Default to python
        const codeContent = languageMatch
          ? code.substring(languageMatch[0].length)
          : code;

        return (
          <SyntaxHighlighter
            key={index}
            language={language}
            style={tomorrow}
            showLineNumbers
            customStyle={{ borderRadius: "0.5rem", padding: "1em" }}
          >
            {codeContent}
          </SyntaxHighlighter>
        );
      } else {
        return <span key={index}>{part}</span>;
      }
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="flex-grow p-6 overflow-auto">
        <div className="flex flex-col space-y-4">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`px-4 py-2 rounded-lg max-w-md lg:max-w-2xl ${
                  msg.isUser
                    ? "bg-blue-500 text-white"
                    : "bg-white text-gray-800"
                }`}
              >
                {renderMessageContent(msg.text)}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="px-4 py-2 rounded-lg bg-white text-gray-800">
                Typing...
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="p-4 bg-white border-t">
        <div className="mb-4 flex flex-wrap gap-2 justify-center">
          {suggestionQuestions.map((question, index) => (
            <button
              key={index}
              onClick={() => handleSuggestionClick(question)}
              className="px-4 py-2 text-sm bg-gray-200 rounded-full hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400"
            >
              {question}
            </button>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-grow px-4 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type your message..."
          />
          <button
            type="submit"
            className="ml-4 px-6 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300"
            disabled={isLoading}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
