import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      if (selectedFile.type !== "video/mp4") {
        setMessage("Only MP4 files are allowed.");
        setFile(null);
      } else {
        setMessage("");
        setFile(selectedFile);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a valid MP4 file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setMessage("Uploading and processing...");

      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(data.message);
      } else {
        setMessage(`Error: ${data.error}`);
      }
    } catch (error) {
      setMessage("Error uploading file. Please try again.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Emotion Detection System</h1>
      <input type="file" accept="video/mp4" onChange={handleFileChange} />
      <br />
      <button
        onClick={handleUpload}
        style={{ marginTop: "10px", padding: "10px", fontSize: "16px" }}
      >
        Upload & Analyze
      </button>
      <p style={{ marginTop: "20px", color: "blue" }}>{message}</p>
    </div>
  );
}

export default App;
