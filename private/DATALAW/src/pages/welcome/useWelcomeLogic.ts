import { useEffect, useState } from 'react';
import { UserService } from '../../services/auth/UserService';
import { createLogger } from '../../services/logger/Logger';
import type { UserProfile } from '../../types/entities';

export type { UserProfile };

export type WelcomeLogic = ReturnType<typeof useWelcomeLogic>;

export function useWelcomeLogic(onSelect: (profile: UserProfile) => void) {
  const log = createLogger('useWelcomeLogic');
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [isGenesis, setIsGenesis] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);
  const [password, setPassword] = useState('');
  const [error, setError] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [adminForm, setAdminForm] = useState({ login: 'admin', full_name: '', password: '' });

  const loadProfiles = async () => {
    setLoading(true);
    try {
      const data = await UserService.listProfiles();
      setProfiles(data);
      setIsGenesis(data.length === 0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProfiles().catch(log.error);
  }, []);

  const handleLogin = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedUser || isVerifying) return;

    setIsVerifying(true);
    setError(false);

    try {
      const profile = await UserService.verifyLogin(selectedUser.id, password);
      if (profile) {
        onSelect(profile);
      } else {
        setError(true);
        setPassword('');
      }
    } finally {
      setIsVerifying(false);
    }
  };

  const handleCreateAdmin = async (event: React.FormEvent) => {
    event.preventDefault();
    await UserService.createInitialSuperadmin(adminForm);
    await loadProfiles();
    setIsGenesis(false);
  };

  return {
    profiles,
    loading,
    isGenesis,
    selectedUser,
    setSelectedUser,
    password,
    setPassword,
    error,
    isVerifying,
    adminForm,
    setAdminForm,
    handleLogin,
    handleCreateAdmin,
  };
}
