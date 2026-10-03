import {defineConfig,devices} from '@playwright/test';
export default defineConfig({testDir:'./browser',fullyParallel:false,workers:1,reporter:'list',timeout:30000,
 use:{baseURL:'http://127.0.0.1:4182'},projects:[{name:'chromium',use:{...devices['Desktop Chrome']}},{name:'webkit-mobile',use:{...devices['iPhone 13']}}],
 webServer:{command:'node bench/server.mjs',url:'http://127.0.0.1:4182',reuseExistingServer:false}});
