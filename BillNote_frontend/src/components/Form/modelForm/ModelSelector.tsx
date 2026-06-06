import { useState, useEffect } from 'react'
import { useModelStore } from '@/store/modelStore'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import toast from 'react-hot-toast'

interface ModelSelectorProps {
  providerId: string
}

export function ModelSelector({ providerId }: ModelSelectorProps) {
  const { models, loading, selectedModel, loadModels, setSelectedModel, addNewModel } =
    useModelStore()
  const [search, setSearch] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [manualInput, setManualInput] = useState('')

  const filteredModels = models.filter(model => {
    const keywords = search.trim().toLowerCase().split(/\s+/)
    const target = model.id.toLowerCase()
    return keywords.every(kw => target.includes(kw))
  })

  useEffect(() => {
    if (providerId) {
      loadModels(providerId)
    }
  }, [providerId])

  const handleSubmit = async () => {
    const modelToSave = selectedModel || manualInput.trim()
    if (!modelToSave) {
      toast.error('请选择或输入一个模型名称')
      return
    }
    try {
      setSubmitting(true)
      await addNewModel(providerId, modelToSave)
      toast.success('保存模型成功 🎉')
      setManualInput('')
    } catch (error) {
      toast.error('保存失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2 font-bold">
        <span>选择模型</span>
        <Button
          variant="ghost"
          type="button"
          onClick={() => loadModels(providerId)}
          disabled={loading}
        >
          {loading ? '加载中...' : '刷新模型'}
        </Button>
      </div>

      <Select value={selectedModel} onValueChange={(val) => { setSelectedModel(val); setManualInput('') }}>
        <SelectTrigger className="w-[300px]">
          <SelectValue placeholder="请选择模型" />
        </SelectTrigger>
        <SelectContent>
          <div className="p-2">
            <Input
              placeholder="搜索模型..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="h-8"
            />
          </div>
          {filteredModels.map((model, index) => (
            <SelectItem key={`${model.id}-${index}`} value={model.id}>
              {model.id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="flex flex-col gap-1">
        <span className="text-sm text-muted-foreground">或手动输入模型名称：</span>
        <Input
          placeholder="输入模型名称，如 claude-opus-4-20250514"
          value={manualInput}
          onChange={e => { setManualInput(e.target.value); setSelectedModel('') }}
          className="w-[300px]"
        />
      </div>

      <Button onClick={handleSubmit} disabled={submitting || (!selectedModel && !manualInput.trim())}>
        {submitting ? '保存中...' : '保存模型'}
      </Button>
    </div>
  )
}
