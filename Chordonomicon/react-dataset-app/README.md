# React Dataset Management App

This project is a React web application designed for managing datasets, performing search testing, and custom exporting of data. It provides a user-friendly interface for dataset operations and is built using TypeScript.

## Features

- **Dataset Management**: Add, edit, and delete datasets with ease.
- **Search Testing**: Search datasets based on user-defined criteria and view results.
- **Custom Exporting**: Export datasets in various formats with configurable options.

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd react-dataset-app
   ```

2. **Install dependencies**:
   ```
   npm install
   ```

3. **Run the application**:
   ```
   npm start
   ```

4. **Open Storybook**:
   ```
   npm run storybook
   ```

## Project Structure

```
react-dataset-app
├── src
│   ├── components
│   │   ├── DatasetManagement
│   │   │   └── DatasetManagement.tsx
│   │   ├── SearchTesting
│   │   │   └── SearchTesting.tsx
│   │   ├── CustomExporting
│   │   │   └── CustomExporting.tsx
│   │   └── index.ts
│   ├── App.tsx
│   └── index.tsx
├── .storybook
│   ├── main.js
│   ├── preview.js
│   └── manager.js
├── package.json
├── tsconfig.json
├── README.md
└── .gitignore
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.