import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  canAssignRole,
  canManageTargetUser,
  normalizeUserRole,
} from '../../services/auth/permissions';
import { UserService } from '../../services/auth/UserService';
import type { UserProfile } from '../../types/entities';
import type { User } from './types';

const DEFAULT_FORM: Partial<User> = {
  full_name: '',
  role: 'LAWYER',
  login: '',
  password: '',
};

export function useUsersLogic(currentUser: UserProfile | null = null) {
  const [rawItems, setRawItems] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'list' | 'analytics'>('list');
  const [selectedItem, setSelectedItem] = useState<User | null>(null);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [formData, setFormData] = useState<Partial<User>>(DEFAULT_FORM);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const users = await UserService.listStaffProfiles();
      setRawItems(users.map((user) => ({ ...user, role: normalizeUserRole(user.role) })));
    } catch (err) {
      // FIX: catch вместо finally-only — предотвращает unhandled rejection
      console.error('[Users] fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchItems();
  }, [fetchItems]);

  const items = useMemo(() => {
    const filtered = rawItems.filter((user) => {
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return user.full_name.toLowerCase().includes(q) || user.login.toLowerCase().includes(q);
    });
    return filtered;
  }, [rawItems, searchQuery]);

  const openCreate = () => {
    setSelectedItem(null);
    setFormData(DEFAULT_FORM);
    setIsEditOpen(true);
  };

  const openEdit = (user: User) => {
    setSelectedItem(user);
    setFormData({ ...user, password: '', _isEdit: true });
    setIsEditOpen(true);
  };

  const openDelete = (user: User) => {
    setSelectedItem(user);
    setIsDeleteOpen(true);
  };

  const closeEdit = () => {
    setSelectedItem(null);
    setIsEditOpen(false);
  };

  const closeDelete = () => {
    setSelectedItem(null);
    setIsDeleteOpen(false);
  };

  const handleSave = async () => {
    if (!currentUser) return;

    if (selectedItem) {
      await UserService.updateUser(currentUser, {
        id: Number(selectedItem.id),
        full_name: String(formData.full_name ?? ''),
        role: String(formData.role ?? 'LAWYER'),
        password: String(formData.password ?? ''),
      });
    } else {
      await UserService.createUser(currentUser, {
        full_name: String(formData.full_name ?? ''),
        login: String(formData.login ?? ''),
        role: String(formData.role ?? 'LAWYER'),
        password: String(formData.password ?? ''),
      });
    }

    await fetchItems();
    closeEdit();
  };

  const confirmDelete = async () => {
    if (!currentUser || !selectedItem) return;
    await UserService.deleteUser(currentUser, Number(selectedItem.id));
    await fetchItems();
    closeDelete();
  };

  return {
    rawItems,
    items,
    loading,
    activeTab,
    setActiveTab,
    selectedItem,
    formData,
    setFormData,
    searchQuery,
    setSearchQuery,
    isEditOpen,
    isDeleteOpen,
    openCreate,
    openEdit,
    openDelete,
    closeEdit,
    closeDelete,
    handleSave,
    confirmDelete,
    canEditUser: (user: User) =>
      canManageTargetUser(currentUser, { id: Number(user.id), role: user.role }),
    canDeleteUser: (user: User) =>
      canManageTargetUser(currentUser, { id: Number(user.id), role: user.role }),
    canAssignRole: (role: string) => canAssignRole(currentUser, role),
  };
}
