import React, { useState } from 'react';
import './App.css';
import { MIDIExporter } from './components/MIDIExporter/MIDIExporter';
import { DatasetManagement } from './components/DatasetManagement/DatasetManagement';
import { SearchTesting } from './components/SearchTesting/SearchTesting';
import { CustomExporting } from './components/CustomExporting/CustomExporting';

function App() {
  // State to track which component is currently active
  const [activeComponent, setActiveComponent] = useState<string>('dataset');

  // Function to render the active component
  const renderActiveComponent = () => {
    switch (activeComponent) {
      case 'dataset':
        return <DatasetManagement />;
      case 'search':
        return <SearchTesting />;
      case 'export':
        return <CustomExporting />;
      case 'midi':
        return <MIDIExporter />;
      default:
        return <DatasetManagement />;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Chordonomicon Dataset Manager</h1>
      </header>

      <nav className="App-nav">
        <ul>
          <li>
            <button 
              className={activeComponent === 'dataset' ? 'active' : ''}
              onClick={() => setActiveComponent('dataset')}
            >
              Dataset Management
            </button>
          </li>
          <li>
            <button 
              className={activeComponent === 'search' ? 'active' : ''}
              onClick={() => setActiveComponent('search')}
            >
              Search & Testing
            </button>
          </li>
          <li>
            <button 
              className={activeComponent === 'export' ? 'active' : ''}
              onClick={() => setActiveComponent('export')}
            >
              Custom Export
            </button>
          </li>
          <li>
            <button 
              className={activeComponent === 'midi' ? 'active' : ''}
              onClick={() => setActiveComponent('midi')}
            >
              MIDI Generator
            </button>
          </li>
        </ul>
      </nav>

      <main>
        {renderActiveComponent()}
      </main>

      <footer className="App-footer">
        <p>Chordonomicon Dataset Manager &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;