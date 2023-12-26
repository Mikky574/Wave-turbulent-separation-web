import { JSX, render } from 'preact'
import './style.css'
import { signal } from '@preact/signals'
import {
  Configuration,
  DefaultApi,
  WsData,
  WsDataToJSON,
} from './_generated/typescript-fetch'

import { NextUIProvider, Button } from '@nextui-org/react'

const u2 = signal<File | null>(null)
const v2 = signal<File | null>(null)
const w2 = signal<File | null>(null)
const theta = signal<File | null>(null)
// const beta = signal<File | null>(null)
const Fs = signal<string>('')
const Fw1 = signal<string>('')
const Fw2 = signal<string>('')
const uploadResult = signal<{ [key: string]: string } | null>(null)
const processResult = signal<string>('')

const App = () => (
  <div className='max-w-sm p-6 bg-white border border-gray-200 rounded-lg shadow text-black'>
    <form onSubmit={onSubmit} className='grid gap-y-3'>
      <label className='grid grid-cols-4'>
        u2
        <input
          type='file'
          required
          className='col-span-3'
          onChange={e => onFileSelected(e, u2)}
        />
      </label>
      <label className='grid grid-cols-4'>
        v2
        <input
          type='file'
          required
          className='col-span-3'
          onChange={e => onFileSelected(e, v2)}
        />
      </label>
      <label className='grid grid-cols-4'>
        w2
        <input
          type='file'
          required
          className='col-span-3'
          onChange={e => onFileSelected(e, w2)}
        />
      </label>
      <label className='grid grid-cols-4'>
        theta
        <input
          type='file'
          required
          className='col-span-3'
          onChange={e => onFileSelected(e, theta)}
        />
      </label>
      {/* <label class='grid grid-cols-4'>
        beta
        <input
          type='file'
          required
          className='col-span-3'
          onChange={e => onFileSelected(e, beta)}
        />
      </label> */}

      <label className='grid grid-cols-4'>
        Fs
        <input
          type='number'
          step='0.0001'
          required
          className='h-8 col-span-3 rounded-lg border border-gray-300'
          value={Fs}
          onInput={e => (Fs.value = e.currentTarget.value)}
        />
      </label>
      <label className='grid grid-cols-4'>
        Fw1
        <input
          type='number'
          step='0.0001'
          required
          className='h-8 col-span-3 rounded-lg border border-gray-300'
          value={Fw1}
          onInput={e => (Fw1.value = e.currentTarget.value)}
        />
      </label>
      <label className='grid grid-cols-4'>
        Fw2
        <input
          type='number'
          step='0.0001'
          required
          className='h-8 col-span-3 rounded-lg border border-gray-300'
          value={Fw2}
          onInput={e => (Fw2.value = e.currentTarget.value)}
        />
      </label>
      <Button
        type='submit'
        isDisabled={
          !u2.value ||
          !v2.value ||
          !w2.value ||
          !theta.value ||
          !Fs.value ||
          !Fw1.value ||
          !Fw2.value
        }
      >
        Submit
      </Button>
    </form>
    {uploadResult.value && (
      <code className='block overflow-x-auto whitespace-pre text-left'>
        {JSON.stringify(uploadResult.value, null, 1)}
      </code>
    )}
    <br />
    {processResult.value || null}
  </div>
)

function onFileSelected(
  e: JSX.TargetedEvent<HTMLInputElement, Event>,
  sig: typeof u2
) {
  const file = e.currentTarget.files?.[0]
  if (file) sig.value = file
}

const defaultApi = new DefaultApi(new Configuration({ basePath: '' }))
function onSubmit(e: Event) {
  e.preventDefault()
  defaultApi
    .uploadFileUploadPost({
      u2: u2.peek()!,
      v2: v2.peek()!,
      w2: w2.peek()!,
      theta: theta.peek()!,
    })
    .then(res => {
      uploadResult.value = res
      const ws = new WebSocket(`ws://${location.host}/ws`)
      ws.onopen = () => {
        const wsData: WsData = {
          action: 'start',
          fs: Number(Fs.peek()),
          fw1: Number(Fw1.peek()),
          fw2: Number(Fw2.peek()),
          filePaths: {
            u2: res['u2'],
            v2: res['v2'],
            w2: res['w2'],
            theta: res['theta'],
          },
        }
        ws.send(JSON.stringify(WsDataToJSON(wsData)))
      }
      ws.onmessage = e => {
        const msg = e.data as string
        console.log(msg)
        processResult.value = msg
        if (msg.includes('Processing completed')) ws.close()
      }
      return res
    })
    .catch(err => alert(JSON.stringify(err)))
}

render(
  <NextUIProvider>
    <App />
  </NextUIProvider>,
  document.getElementById('app')!
)
