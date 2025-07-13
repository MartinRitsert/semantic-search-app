# Frontend (Next.js)
This directory contains the Next.js frontend for the semantic search application.

## Prerequisites
*   Node.js (LTS version recommended)
*   pnpm

## Setup
1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install dependencies:**
    ```bash
    pnpm install
    ```
3.  **Create a local environment file:**
    Create a `.env.local` file in this directory. You can leave it empty for now, but it's good practice to have it for any frontend-specific environment variables.
    ```bash
    touch .env.local
    ```
    
## Running the Development Server
To start the development server, run the following command from the `frontend` directory:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.