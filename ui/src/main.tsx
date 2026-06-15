import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

const THEME_STORAGE_KEY = 'evalora-theme';

const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
const preferredTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
const initialTheme = savedTheme === 'dark' || savedTheme === 'light' ? savedTheme : preferredTheme;

document.documentElement.dataset.theme = initialTheme;
document.documentElement.style.colorScheme = initialTheme;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
