-- setup supabase tables untuk portfolio backend
-- jalankan script ini di supabase sql editor

-- 1. table untuk menyimpan embeddings
create table if not exists embeddings (
    id bigserial primary key,
    document_id text not null,
    embedding jsonb not null,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- 2. table untuk menyimpan conversations
create table if not exists conversations (
    id bigserial primary key,
    session_id text not null,
    question text not null,
    response text not null,
    message_type text default 'general',
    confidence_score real,
    metadata jsonb,
    created_at timestamp with time zone default now()
);

-- 3. table untuk menyimpan sessions
create table if not exists sessions (
    id bigserial primary key,
    session_id text unique not null,
    session_data jsonb not null,
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now(),
    expires_at timestamp with time zone
);

-- 4. buat indexes untuk performance
create index if not exists idx_conversations_session_id on conversations(session_id);
create index if not exists idx_conversations_created_at on conversations(created_at);
create index if not exists idx_conversations_message_type on conversations(message_type);

create index if not exists idx_sessions_session_id on sessions(session_id);
create index if not exists idx_sessions_expires_at on sessions(expires_at);

create index if not exists idx_embeddings_document_id on embeddings(document_id);

-- 5. row level security (rls) - opsional untuk keamanan
alter table conversations enable row level security;
alter table sessions enable row level security;
alter table embeddings enable row level security;

-- 6. policies untuk akses data (opsional - untuk demo bisa dilewati)
-- create policy "allow all" on conversations for all using (true);
-- create policy "allow all" on sessions for all using (true); 
-- create policy "allow all" on embeddings for all using (true);

-- 7. test table dengan insert dummy data
insert into conversations (session_id, question, response, message_type) 
values ('test-session', 'hello', 'hi there!', 'greeting')
on conflict do nothing;

-- 8. verifikasi tables berhasil dibuat
select 'conversations' as table_name, count(*) as row_count from conversations
union all
select 'sessions' as table_name, count(*) as row_count from sessions  
union all
select 'embeddings' as table_name, count(*) as row_count from embeddings;