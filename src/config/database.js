import { config as dotenvConfig } from 'dotenv';
import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenvConfig();

// Database configuration
const dbConfig = {
    user: process.env.DB_USER || 'solvedbook',
    host: process.env.DB_HOST || 'localhost',
    database: process.env.DB_NAME || 'solvedbook',
    password: process.env.DB_PASSWORD || 'password',
    port: process.env.DB_PORT || 5432,
};

let pool;

const initializeDatabase = async () => {
    try {
        pool = new Pool(dbConfig);
        
        // Test the connection
        const client = await pool.connect();
        console.log('Connected to the PostgreSQL database.');
        client.release();
        
        // Initialize schema if needed
        await initializeSchema();
    } catch (err) {
        console.error('Error connecting to database:', err.message);
        throw err;
    }
};

const initializeSchema = async () => {
    try {
        const client = await pool.connect();
        
        // Check if tables exist
        const result = await client.query(`
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'cases'
            );
        `);
        
        if (!result.rows[0].exists) {
            console.log('Initializing database schema...');
            const schemaPath = path.resolve(__dirname, '../../database/schema.sql');
            const schema = fs.readFileSync(schemaPath, 'utf8');
            await client.query(schema);
            console.log('Database schema initialized.');
        }
        
        client.release();
    } catch (err) {
        console.error('Error initializing schema:', err.message);
        throw err;
    }
};

const getDatabase = () => {
    return pool;
};

const closeDatabase = async () => {
    if (pool) {
        await pool.end();
    }
};

export {
    initializeDatabase,
    getDatabase,
    closeDatabase,
};