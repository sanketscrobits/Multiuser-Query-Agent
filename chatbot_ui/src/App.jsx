
import './App.css'

function App() {

  return (
    <>
      {/* Hero Section */}
      <section className="min-h-screen flex flex-col justify-center items-center bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white text-center px-6">
        <h1 className="text-5xl font-bold mb-4">
          Welcome to <span className="text-blue-400">ScroBits Chat</span>
        </h1>
        <p className="text-gray-300 max-w-xl mb-8">
          Experience a modern AI chatbot interface built with Vite + React + Tailwind.
          Scroll down to explore more sections below 👇
        </p>
        <button className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-full text-lg font-semibold transition duration-200">
          Get Started
        </button>
      </section>

      {/* Features Section */}
      <section className="min-h-screen bg-white flex flex-col justify-center items-center text-gray-800 px-6">
        <h2 className="text-4xl font-bold mb-6 text-blue-500">Features</h2>
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl">
          <div className="p-6 border rounded-xl shadow-md hover:shadow-xl transition">
            <i className="fa-solid fa-bolt text-blue-600 text-3xl mb-4"></i>
            <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
            <p>Powered by Vite and React for an ultra-fast experience.</p>
          </div>
          <div className="p-6 border rounded-xl shadow-md hover:shadow-xl transition">
            <i className="fa-solid fa-robot text-blue-600 text-3xl mb-4"></i>
            <h3 className="text-xl font-semibold mb-2">AI-Driven</h3>
            <p>Integrates with AI APIs like OpenAI and OpenRouter for intelligent responses.</p>
          </div>
          <div className="p-6 border rounded-xl shadow-md hover:shadow-xl transition">
            <i className="fa-solid fa-shield-halved text-blue-600 text-3xl mb-4"></i>
            <h3 className="text-xl font-semibold mb-2">Secure</h3>
            <p>Data-safe architecture with clean modular backend integration.</p>
          </div>
        </div>
      </section>
      
    </>
  )
}

export default App
