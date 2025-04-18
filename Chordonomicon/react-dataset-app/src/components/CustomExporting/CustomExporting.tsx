import React, { useState } from 'react';

const CustomExporting: React.FC = () => {
    const [selectedDataset, setSelectedDataset] = useState<string>('');
    const [exportFormat, setExportFormat] = useState<string>('csv');

    const handleExportMIDI = async () => {
        try {
            const response = await fetch('/api/export-midi', {
                method: 'POST',
            });
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'generated_song.mid';
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error exporting MIDI file:', error);
        }
    };

    const handleDatasetChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedDataset(event.target.value);
    };

    const handleFormatChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setExportFormat(event.target.value);
    };

    const handleExport = () => {
        // Logic for exporting the dataset in the selected format
        console.log(`Exporting dataset: ${selectedDataset} in format: ${exportFormat}`);
    };

    return (
        <div>
            <h2>Custom Exporting</h2>
            <label htmlFor="dataset-select">Select Dataset:</label>
            <select id="dataset-select" value={selectedDataset} onChange={handleDatasetChange}>
                <option value="">--Select a dataset--</option>
                {/* Add options for datasets here */}
            </select>

            <label htmlFor="format-select">Select Export Format:</label>
            <select id="format-select" value={exportFormat} onChange={handleFormatChange}>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
                <option value="xml">XML</option>
                {/* Add more formats as needed */}
            </select>

            <button onClick={handleExport}>Export</button>
            <button onClick={handleExportMIDI}>Export as MIDI</button>
        </div>
    );
};

export default CustomExporting;

const CustomExporting: React.FC = () => {
    const [selectedDataset, setSelectedDataset] = useState<string>('');
    const [exportFormat, setExportFormat] = useState<string>('csv');

    const handleDatasetChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedDataset(event.target.value);
    };

    const handleFormatChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setExportFormat(event.target.value);
    };

    const handleExport = () => {
        // Logic for exporting the dataset in the selected format
        console.log(`Exporting dataset: ${selectedDataset} in format: ${exportFormat}`);
    };

    return (
        <div>
            <h2>Custom Exporting</h2>
            <label htmlFor="dataset-select">Select Dataset:</label>
            <select id="dataset-select" value={selectedDataset} onChange={handleDatasetChange}>
                <option value="">--Select a dataset--</option>
                {/* Add options for datasets here */}
            </select>

            <label htmlFor="format-select">Select Export Format:</label>
            <select id="format-select" value={exportFormat} onChange={handleFormatChange}>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
                <option value="xml">XML</option>
                {/* Add more formats as needed */}
            </select>

            <button onClick={handleExport}>Export</button>
        </div>
    );
};

export default CustomExporting;